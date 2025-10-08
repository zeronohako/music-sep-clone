import argparse, json, os, numpy as np, soundfile as sf, musdb
from src.dsp.stft import wave_to_spec, spec_to_wave
from src.dsp.masks import STEMS, ideal_ratio_masks, apply_mask
from src.eval.museval_wrap import score_track
from time import perf_counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musdb_root", required=True)
    ap.add_argument("--subset", default="test", choices=["train","test"])
    ap.add_argument("--out", default="experiments/results/baselines.json")
    ap.add_argument("--write_wavs", action="store_true")
    ap.add_argument("--mwf", action="store_true", help="Apply MWF to BU baseline")
    ap.add_argument("--mwf_iter", type=int, default=1, help="EM iterations for MWF (1 is typical)")
    ap.add_argument("--max_tracks", type=int, default=1, help="Limit number of tracks processed (0 = all)")
    ap.add_argument("--max_seconds", type=float, default=10.0, help="Crop each track to first N seconds (0 = full)")
    ap.add_argument("--skip_eval", action="store_true", help="Build estimates (and WAVs) but skip museval scoring")

    args = ap.parse_args()

    mus = musdb.DB(root=args.musdb_root, subsets=args.subset)
    results = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for track in mus:  # each is a MUSDB Track
        processed = 0
        for track in mus:  # each is a MUSDB Track
            processed += 1
            if args.max_tracks and processed > args.max_tracks:
                break
        
        t0 = perf_counter()
        mix = track.audio.astype(np.float32)           # (T,2)
        mix = track.audio.astype(np.float32)
        refs = {s: track.targets[s].audio.astype(np.float32) for s in STEMS}
        sr = getattr(track, "rate", 44100)
        if args.max_seconds and args.max_seconds > 0:
            nsamp = int(sr * args.max_seconds)
            mix = mix[:nsamp]
            for s in STEMS:
                refs[s] = refs[s][:nsamp]
        t1 = perf_counter()
        mix_S, mix_mag, _ = wave_to_spec(mix)
        t2 = perf_counter()

        # BU: Ideal Ratio Mask baseline (magnitudes -> wav)
        ref_mags = {}
        for s in STEMS:
            S_ref, mag_ref, _ = wave_to_spec(refs[s])
            ref_mags[s] = mag_ref
        irm = ideal_ratio_masks(ref_mags)
        t3 = perf_counter()

        est_BU = {}
        for s in STEMS:
            S_hat = apply_mask(mix_S, irm[s])
            est_BU[s] = spec_to_wave(S_hat)
        t4 = perf_counter()

        # BU + MWF: build per-stem magnitude guesses from IRM * mixture mag, then MWF
        if args.mwf:
            from src.post.mwf import apply_mwf
            # estimated mags BEFORE MWF come from masking the mixture magnitude
            est_mags = {s: irm[s] * np.abs(mix_S) for s in STEMS}  # (2,F,T) each
            Y = apply_mwf(mix_S, est_mags, iterations=args.mwf_iter)
            est_BU_MWF = {s: spec_to_wave(Y[s]) for s in STEMS}
            scores_BU_MWF = score_track(refs, est_BU_MWF)
        t5 = perf_counter()

        # BL: Mixture-as-estimate baseline
        est_BL = {s: mix.copy() for s in STEMS}
        
        # score
        scores_BU = score_track(refs, est_BU)
        scores_BL = score_track(refs, est_BL)

        results[track.name] = {"BL": scores_BL, "BU": scores_BU}
        if args.mwf:
            results[track.name]["BU_MWF"] = scores_BU_MWF
        t6 = perf_counter()

        print(
            f"[{track.name}] "
            f"I/O={(t1-t0):.3f}s  STFT={(t2-t1):.3f}s  IRM={(t3-t2):.3f}s  "
            f"ISTFT={(t4-t3):.3f}s  MWF={(t5-t4):.3f}s  EVAL={(t6-t5):.3f}s"
        )

        # score (optional)
        if not args.skip_eval:
            scores_BU = score_track(refs, est_BU)
            scores_BL = score_track(refs, est_BL)
            results[track.name] = {"BL": scores_BL, "BU": scores_BU}
            if args.mwf:
                results[track.name]["BU_MWF"] = scores_BU_MWF
            else:
                results[track.name] = {"note": "skip_eval=True (no museval run)"}

        if args.write_wavs:
            base = os.path.join("experiments", "results", "wavs", track.name)
            os.makedirs(base, exist_ok=True)
            for s in STEMS:
                sf.write(os.path.join(base, f"BU_{s}.wav"), est_BU[s], samplerate=44100)
                sf.write(os.path.join(base, f"BL_{s}.wav"), est_BL[s], samplerate=44100)
            if args.mwf:
                for s in STEMS:
                    sf.write(os.path.join(base, f"BU_MWF_{s}.wav"), est_BU_MWF[s], samplerate=44100)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
