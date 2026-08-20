#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse on-board msprof op Source dumps into per-file / per-line instruction-execution counts.

Input  : a dump directory produced by `msprof op --aic-metrics=...Source` (ccec_g build),
         containing fdata, kernel0Stub.o.bbbmap.0, aicore_binary.o, op_basic_info.txt.
Output : per_file.csv, per_line.csv, summary.txt next to the dump dir.

Mechanism:
  fdata line:  `1 <kernel_name> <pc_offset_hex> <exec_count>`  (no_lbr bbcount)
  Use CANN addr2line (aarch64-target-linux-gnu-addr2line) to map pc_offset -> (func, file:line).
  Aggregate exec_count by source file and by (file,line).

NOTE: counts are basic-block execution counts (instrumented by ccec_g). They are a proxy for
total instruction executions, NOT cycle counts. Relative comparison across two operators built
and run identically is the intended use.
"""
import os, sys, csv, re, subprocess, argparse

def cann_addr2line():
    p = "/usr/local/Ascend/cann-8.5.1/tools/hcc/bin/aarch64-target-linux-gnu-addr2line"
    if os.path.exists(p):
        return p
    # fallback search
    for cand in ["/usr/local/Ascend/ascend-toolkit/latest/tools/hcc/bin/aarch64-target-linux-gnu-addr2line"]:
        if os.path.exists(cand):
            return cand
    return "addr2line"

def parse_fdata(path):
    """Return list of (kernel_name, pc_offset_hex, exec_count)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("no_lbr"):
                continue
            parts = line.split()
            if len(parts) < 4 or parts[0] != "1":
                continue
            kname, pc, cnt = parts[1], parts[2], parts[3]
            try:
                cnt_i = int(cnt)
            except ValueError:
                continue
            rows.append((kname, pc, cnt_i))
    return rows

def addr2line_batch(binary, pcs):
    """Run addr2line once feeding all pcs on stdin. Returns dict pc -> (func, file, line)."""
    al = cann_addr2line()
    # addr2line prints two lines per address (function, then file:line:col) when -f is given.
    proc = subprocess.run(
        [al, "-e", binary, "-f", "-C"],
        input="\n".join(pcs) + "\n",
        capture_output=True, text=True
    )
    out = proc.stdout.split("\n")
    res = {}
    # pairs: func, then "file:line:col" (or "file:?" or "??")
    i = 0
    for idx, pc in enumerate(pcs):
        func = out[i].strip() if i < len(out) else "??"
        loc = out[i+1].strip() if i+1 < len(out) else "??"
        i += 2
        file, line = "", ""
        if loc and loc != "??":
            # file:line:col  OR  file:?
            m = re.match(r"^(.*):(\d+)(?::\d+)?$", loc)
            if m:
                file, line = m.group(1), m.group(2)
            elif loc.endswith(":?"):
                file = loc[:-2]
                line = "?"
            else:
                file = loc
        res[pc] = (func, file, line)
    return res

def normalize_file(path):
    """Map a build/CANN path to a short, merge-stable label."""
    if not path:
        return "<unknown>"
    # vllm-ascend new TQ FIA op (turboquant_fia_mse8bit)
    idx = path.find("turboquant_fia_mse8bit/")
    if idx != -1:
        return "csrc/" + path[idx:]
    idx = path.find("op_kernel/vendored/arch32/")
    if idx != -1:
        return "csrc/turboquant_fia_mse8bit/" + path[idx:]
    # legacy ops-transformer paths (if profiling old builds)
    idx = path.find("op_kernel/arch32/")
    if idx != -1:
        tail = path[idx:]
        return "attention/common/" + tail
    idx = path.find("common/op_kernel/")
    if idx != -1:
        return "attention/" + path[idx:]
    # tiling_data / kernel.cpp under kernel_meta
    if "/kernel_meta_" in path or path.endswith("_kernel.cpp") or path.endswith("_tiling_data.h"):
        return "gen/kernel_meta/" + os.path.basename(path)
    # CANN headers
    if "/Ascend/" in path or "/asc/impl/" in path or "/bisheng_compiler/" in path or "/clang/" in path:
        return "cann/" + os.path.basename(path)
    # generated src
    if "/build/binary/" in path:
        return "build/" + path.split("/build/binary/")[1]
    return os.path.basename(path)

def is_pipe(kname):
    if kname.endswith("_aiv"):
        return "aiv"
    if kname.endswith("_aic"):
        return "aic"
    return "?"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="OPPROF_*/dump directory")
    ap.add_argument("--label", default="op", help="label for output files")
    args = ap.parse_args()

    dd = args.dump_dir.rstrip("/")
    fdata = os.path.join(dd, "fdata")
    binary = os.path.join(dd, "aicore_binary.o")
    if not os.path.exists(fdata):
        sys.exit("no fdata in " + dd)
    if not os.path.exists(binary):
        sys.exit("no aicore_binary.o in " + dd)

    rows = parse_fdata(fdata)
    if not rows:
        sys.exit("fdata empty: " + fdata)

    # unique pcs (hex strings, addr2line handles them)
    uniq = sorted(set(r[1] for r in rows), key=lambda x: int(x, 16))
    print(f"[*] {len(uniq)} unique PCs, {len(rows)} fdata rows", file=sys.stderr)
    mapping = addr2line_batch(binary, uniq)

    # op_basic_info
    bidi, dur = "", ""
    obi = os.path.join(dd, "op_basic_info.txt")
    if os.path.exists(obi):
        for ln in open(obi):
            if ln.startswith("Block Dim="):
                bidi = ln.split("=",1)[1].strip()
            if ln.startswith("Task Duration"):
                dur = ln.split("=",1)[1].strip()
    # also read OpBasicInfo.csv for Task Duration (op_basic_info lacks it on op mode)
    csv_p = os.path.join(os.path.dirname(dd), "OpBasicInfo.csv")
    if os.path.exists(csv_p) and not dur:
        try:
            with open(csv_p) as f:
                rd = list(csv.reader(f))
            if len(rd) > 1 and len(rd[0]) > 2:
                dur = rd[1][2]  # Task Duration(us)
        except Exception:
            pass

    # aggregate
    per_file = {}   # (pipe, file) -> [total, n_pcs]
    per_line = {}   # (pipe, file, line, func) -> total
    pipe_total = {"aic": 0, "aiv": 0, "?": 0}
    n_pcs_pipe = {"aic": 0, "aiv": 0, "?": 0}
    for kname, pc, cnt in rows:
        pipe = is_pipe(kname)
        func, file, line = mapping.get(pc, ("??", "", "?"))
        nf = normalize_file(file)
        key = (pipe, nf)
        v = per_file.setdefault(key, [0, 0])
        v[0] += cnt; v[1] += 1
        per_line[(pipe, nf, line, func)] = per_line.get((pipe, nf, line, func), 0) + cnt
        pipe_total[pipe] += cnt
        n_pcs_pipe[pipe] += 1

    out_dir = dd.replace("/dump", "")  # OPPROF_xxx/
    base = os.path.join(out_dir, f"{args.label}")
    with open(base + ".per_file.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipe", "file", "exec_count_total", "n_pcs", "exec_count_pct_of_pipe"])
        for (pipe, nf), (tot, npcs) in sorted(per_file.items(), key=lambda kv: kv[1][0], reverse=True):
            denom = pipe_total.get(pipe, 1)
            w.writerow([pipe, nf, tot, npcs, f"{100.0*tot/denom:.2f}"])
    with open(base + ".per_line.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipe", "file", "line", "func", "exec_count"])
        for (pipe, nf, line, func), tot in sorted(per_line.items(), key=lambda kv: kv[1], reverse=True):
            w.writerow([pipe, nf, line, func, tot])
    with open(base + ".summary.txt", "w") as f:
        f.write(f"dump_dir: {dd}\n")
        f.write(f"label: {args.label}\n")
        f.write(f"Block Dim: {bidi}\n")
        f.write(f"Task Duration(us): {dur}\n")
        f.write(f"unique PCs: {len(uniq)}\n")
        f.write(f"fdata rows: {len(rows)}\n")
        for p in ("aic", "aiv"):
            f.write(f"  {p}: total_exec_count={pipe_total[p]}  nonzero_pcs={n_pcs_pipe[p]}\n")
        f.write(f"  grand total exec_count(aic+aiv)={pipe_total['aic']+pipe_total['aiv']}\n")
    print(f"[+] wrote {base}.per_file.csv / .per_line.csv / .summary.txt", file=sys.stderr)

if __name__ == "__main__":
    main()
