import random
from collections import defaultdict

from torch.utils.data import Sampler

from .data_pipeline import normalize_text, safe_int


class TranscriptGroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=16, group_size=4, samples_per_group=4, seed=42, drop_last=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.samples_per_group = int(samples_per_group)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        if self.batch_size != self.group_size * self.samples_per_group:
            raise ValueError("batch_size must equal group_size * samples_per_group for repeated transcript batches.")

        buckets = defaultdict(list)
        for i, sid in enumerate([safe_int(v, -1) for v in dataset["supcon_id"]]):
            if sid != -1:
                buckets[sid].append(i)

        self.buckets = dict(buckets)
        self.keys = [k for k, idxs in self.buckets.items() if len(idxs) >= 2]
        if len(self.keys) == 0:
            raise ValueError("No transcript bucket has >=2 samples; cannot form SupCon positives.")

        if self.drop_last:
            self._len = len(self.keys) // max(1, self.group_size)
        else:
            self._len = max(1, (len(self.keys) + self.group_size - 1) // max(1, self.group_size))

    def __len__(self):
        return self._len

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        keys = self.keys[:]
        rng.shuffle(keys)

        batch = []
        for k in keys:
            idxs = self.buckets[k]
            if len(idxs) >= self.samples_per_group:
                chosen = rng.sample(idxs, self.samples_per_group)
            else:
                chosen = [rng.choice(idxs) for _ in range(self.samples_per_group)]

            batch.extend(chosen)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if (not self.drop_last) and len(batch) > 0:
            yield batch


class TranscriptAlternateBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=16, group_size=4, samples_per_group=4, ctc_batches_per_supcon=1, 
                 include_singletons_as_ctc=True, seed=42, drop_last=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.samples_per_group = int(samples_per_group)
        self.ctc_batches_per_supcon = int(max(1, ctc_batches_per_supcon))
        self.include_singletons_as_ctc = bool(include_singletons_as_ctc)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        sup_ids = [safe_int(v, -1) for v in dataset["supcon_id"]]
        buckets = defaultdict(list)
        ctc_only = []

        for i, sid in enumerate(sup_ids):
            if sid == -1:
                ctc_only.append(i)
            else:
                buckets[sid].append(i)

        self.supcon_buckets = dict(buckets)
        self.supcon_keys = [k for k, idxs in self.supcon_buckets.items() if len(idxs) >= 2]
        if len(self.supcon_keys) == 0:
            raise ValueError("No supcon_id bucket has >=2 samples; cannot build SupCon batches.")

        if self.include_singletons_as_ctc:
            for _, idxs in self.supcon_buckets.items():
                if len(idxs) == 1:
                    ctc_only.extend(idxs)

        self.ctc_only_indices = ctc_only
        self.all_indices = list(range(len(self.dataset)))

        if self.drop_last:
            self._len = max(1, len(self.dataset) // max(1, self.batch_size))
        else:
            self._len = max(1, (len(self.dataset) + self.batch_size - 1) // self.batch_size)

    def __len__(self):
        return self._len

    def _sample_from_pool(self, rng, pool, n):
        if len(pool) == 0:
            return []
        if len(pool) >= n:
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]

    def _build_supcon_batch(self, rng):
        batch = []
        for _ in range(self.group_size):
            k = rng.choice(self.supcon_keys)
            idxs = self.supcon_buckets[k]
            batch.extend(self._sample_from_pool(rng, idxs, self.samples_per_group))

        while len(batch) < self.batch_size:
            k = rng.choice(self.supcon_keys)
            batch.append(rng.choice(self.supcon_buckets[k]))
        return batch[: self.batch_size]

    def _build_ctc_batch(self, rng):
        pool = self.ctc_only_indices if len(self.ctc_only_indices) > 0 else self.all_indices
        batch = self._sample_from_pool(rng, pool, self.batch_size)
        while len(batch) < self.batch_size:
            batch.append(rng.choice(self.all_indices))
        return batch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        produced = 0
        phase = 0
        while produced < self._len:
            batch = self._build_supcon_batch(rng) if phase == 0 else self._build_ctc_batch(rng)
            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                yield batch
                produced += 1
            phase = (phase + 1) % (self.ctc_batches_per_supcon + 1)


class SyntheticAlternateBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=16, group_size=4, samples_per_group=4, mixed_batches=1, ctc_only_batches=1, seed=42, drop_last=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.samples_per_group = int(samples_per_group)
        self.mixed_batches = int(max(1, mixed_batches))
        self.ctc_only_batches = int(max(1, ctc_only_batches))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        if self.samples_per_group != 4:
            raise ValueError("This sampler requires samples_per_group=4 (1 orig + 3 synth).")

        trans = list(dataset["transcript"])
        is_synth = [safe_int(x, 0) for x in dataset["is_synth"]]
        tts = list(dataset["TTS"])
        src = list(dataset["source"])

        self.orig_by_t = defaultdict(list)
        self.synth_by_t = defaultdict(list)
        self.orig_indices = []

        for i, t in enumerate(trans):
            if is_synth[i] == 0:
                self.orig_by_t[t].append(i)
                self.orig_indices.append(i)
            else:
                self.synth_by_t[t].append((i, tts[i], src[i]))

        self.eligible_t = [t for t in self.orig_by_t if len(self.synth_by_t.get(t, [])) >= 3]
        if len(self.eligible_t) == 0:
            raise ValueError("No transcript has both originals and at least 3 synthetic samples.")

        eligible_set = set(self.eligible_t)
        self.exclusive_orig_indices = [i for t, idxs in self.orig_by_t.items() if t not in eligible_set for i in idxs]
        if len(self.exclusive_orig_indices) == 0:
            self.exclusive_orig_indices = self.orig_indices[:]

        if self.drop_last:
            self._len = max(1, len(self.dataset) // max(1, self.batch_size))
        else:
            self._len = max(1, (len(self.dataset) + self.batch_size - 1) // self.batch_size)

    def __len__(self):
        return self._len

    def _pick_three_synth(self, candidates, rng):
        cand = candidates[:]
        rng.shuffle(cand)

        chosen = []
        used_tts = set()
        used_src = set()
        for idx, tts, src in cand:
            if tts in used_tts or src in used_src:
                continue
            chosen.append(idx)
            used_tts.add(tts)
            used_src.add(src)
            if len(chosen) == 3:
                return chosen

        chosen = []
        used_tts = set()
        for idx, tts, _ in cand:
            if tts in used_tts:
                continue
            chosen.append(idx)
            used_tts.add(tts)
            if len(chosen) == 3:
                return chosen

        return [cand[i % len(cand)][0] for i in range(3)]

    def _sample_pool(self, pool, n, rng):
        if len(pool) >= n:
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]

    def _build_mixed_batch(self, rng):
        batch = []
        for _ in range(self.group_size):
            t = rng.choice(self.eligible_t)
            orig_idx = rng.choice(self.orig_by_t[t])
            synth_idxs = self._pick_three_synth(self.synth_by_t[t], rng)
            batch.extend([orig_idx] + synth_idxs)

        while len(batch) < self.batch_size:
            batch.append(rng.choice(self.orig_indices))
        return batch[: self.batch_size]

    def _build_ctc_only_batch(self, rng):
        return self._sample_pool(self.exclusive_orig_indices, self.batch_size, rng)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        cycle = ["mixed"] * self.mixed_batches + ["ctc"] * self.ctc_only_batches
        produced = 0
        phase = 0

        while produced < self._len:
            kind = cycle[phase]
            batch = self._build_mixed_batch(rng) if kind == "mixed" else self._build_ctc_only_batch(rng)
            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                yield batch
                produced += 1
            phase = (phase + 1) % len(cycle)


class HybridAlternateBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=24, group_size=6, samples_per_group=4, mode_schedule=None, seed=42, drop_last=True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.samples_per_group = int(samples_per_group)
        self.mode_schedule = mode_schedule or {"sim": 1, "tts": 1, "ctc": 1}
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        trans = [normalize_text(t) for t in dataset["transcript"]]
        is_synth = [safe_int(x, 0) for x in dataset["is_synth"]]
        mode = list(dataset["mode"])
        sup_ids = [safe_int(x, -1) for x in dataset["supcon_id"]]
        tts = list(dataset["TTS"])
        src = list(dataset["source"])

        self.all_orig_idx = [i for i, s in enumerate(is_synth) if s == 0]

        sim_buckets = defaultdict(list)
        for i, (m, s, sid) in enumerate(zip(mode, is_synth, sup_ids)):
            if s == 0 and m == "sim" and sid != -1:
                sim_buckets[sid].append(i)
        self.sim_buckets = {k: v for k, v in sim_buckets.items() if len(v) >= 2}
        self.sim_keys = list(self.sim_buckets.keys())

        self.tts_orig_by_t = defaultdict(list)
        self.tts_synth_by_t = defaultdict(list)
        for i, (m, s, t) in enumerate(zip(mode, is_synth, trans)):
            if m != "tts":
                continue
            if s == 0:
                self.tts_orig_by_t[t].append(i)
            else:
                self.tts_synth_by_t[t].append((i, tts[i], src[i]))
        self.tts_keys = [t for t in self.tts_orig_by_t if len(self.tts_synth_by_t.get(t, [])) >= 3]

        self.ctc_orig_idx = [i for i, (m, s) in enumerate(zip(mode, is_synth)) if s == 0 and m == "ctc"]
        if len(self.ctc_orig_idx) == 0:
            self.ctc_orig_idx = self.all_orig_idx[:]

        cleaned = {}
        for k, v in self.mode_schedule.items():
            if v <= 0:
                continue
            if k == "sim" and len(self.sim_keys) == 0:
                continue
            if k == "tts" and len(self.tts_keys) == 0:
                continue
            if k == "ctc" and len(self.ctc_orig_idx) == 0:
                continue
            cleaned[k] = int(v)
        if len(cleaned) == 0:
            cleaned = {"ctc": 1}
        self.mode_schedule = cleaned

        if self.drop_last:
            self._len = max(1, len(self.dataset) // max(1, self.batch_size))
        else:
            self._len = max(1, (len(self.dataset) + self.batch_size - 1) // self.batch_size)

    def __len__(self):
        return self._len

    def _sample_pool(self, pool, n, rng):
        if len(pool) == 0:
            return []
        if len(pool) >= n:
            return rng.sample(pool, n)
        return [rng.choice(pool) for _ in range(n)]

    def _pick_three_synth(self, candidates, rng):
        cand = candidates[:]
        rng.shuffle(cand)

        chosen = []
        used_tts = set()
        used_src = set()
        for idx, tts_name, src_name in cand:
            if tts_name in used_tts or src_name in used_src:
                continue
            chosen.append(idx)
            used_tts.add(tts_name)
            used_src.add(src_name)
            if len(chosen) == 3:
                return chosen

        chosen = []
        used_tts = set()
        for idx, tts_name, _ in cand:
            if tts_name in used_tts:
                continue
            chosen.append(idx)
            used_tts.add(tts_name)
            if len(chosen) == 3:
                return chosen

        return [cand[i % len(cand)][0] for i in range(3)]

    def _build_sim_batch(self, rng):
        batch = []
        for _ in range(self.group_size):
            sid = rng.choice(self.sim_keys)
            idxs = self.sim_buckets[sid]
            if len(idxs) >= self.samples_per_group:
                chosen = rng.sample(idxs, self.samples_per_group)
            else:
                chosen = [rng.choice(idxs) for _ in range(self.samples_per_group)]
            batch.extend(chosen)

        while len(batch) < self.batch_size:
            batch.append(rng.choice(self.all_orig_idx))
        return batch[: self.batch_size]

    def _build_tts_batch(self, rng):
        batch = []
        for _ in range(self.group_size):
            t = rng.choice(self.tts_keys)
            orig_idx = rng.choice(self.tts_orig_by_t[t])
            synth_idxs = self._pick_three_synth(self.tts_synth_by_t[t], rng)
            batch.extend([orig_idx] + synth_idxs)

        while len(batch) < self.batch_size:
            pool = self.ctc_orig_idx if len(self.ctc_orig_idx) > 0 else self.all_orig_idx
            batch.append(rng.choice(pool))
        return batch[: self.batch_size]

    def _build_ctc_batch(self, rng):
        return self._sample_pool(self.ctc_orig_idx, self.batch_size, rng)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        cycle = [k for k, v in self.mode_schedule.items() for _ in range(v)]
        produced = 0
        phase = 0

        while produced < self._len:
            mode = cycle[phase]
            if mode == "sim":
                batch = self._build_sim_batch(rng)
            elif mode == "tts":
                batch = self._build_tts_batch(rng)
            else:
                batch = self._build_ctc_batch(rng)

            if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                yield batch
                produced += 1

            phase = (phase + 1) % len(cycle)
