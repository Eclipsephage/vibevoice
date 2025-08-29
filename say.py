import os, sys, argparse, subprocess, shlex, pathlib
os.environ.setdefault("TRANSFORMERS_NO_CUDA","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
os.environ.setdefault("USE_FLASH_ATTENTION_2","0")
os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE","1")
os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION","sdpa")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="microsoft/VibeVoice-1.5B")
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default="en-Alice_woman")
    ap.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    ap.add_argument("--out", default="outputs/out.wav")
    args = ap.parse_args()
    pathlib.Path("demo/text_examples").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)
    txt_path = "demo/text_examples/one_line.txt"
    with open(txt_path,"w") as f:
        f.write(args.text)
    cmd = f"{shlex.quote(sys.executable)} demo/inference_from_file.py --model_path {shlex.quote(args.model_path)} --txt_path {shlex.quote(txt_path)} --speaker_names {shlex.quote(args.speaker)} --device {shlex.quote(args.device)}"
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    outs = sorted(pathlib.Path('outputs').glob('*.wav'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not outs:
        sys.exit('No WAV produced')
    last = outs[0]
    if args.out != str(last):
        pathlib.Path(args.out).write_bytes(last.read_bytes())
    print("Saved:", args.out)
if __name__ == "__main__":
    main()
