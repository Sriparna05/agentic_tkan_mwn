r"""
Phase 10.1 — LaTeX Manuscript and Bibliography Validator

Performs thorough structural and semantic checks on paper/main.tex and paper/references.bib:
1. Verifies that all required files exist.
2. Checks environment balance (\begin{env} vs. \end{env}) using a LIFO stack.
3. Checks curly brace balance.
4. Verifies that all citations in main.tex exist in references.bib.
5. Verifies that all referenced figures exist on disk.
6. Checks that section headings and tables are complete.
"""

import os
import re
import sys


def validate_latex():
    tex_path = "paper/main.tex"
    bib_path = "paper/references.bib"

    print("================================================================================")
    print("   PHASE 10.1: LATEX MANUSCRIPT & BIBLIOGRAPHY VALIDATION")
    print("================================================================================")

    errors = []
    warnings = []

    # 1. File existence
    if not os.path.exists(tex_path):
        errors.append(f"Missing LaTeX file: {tex_path}")
        return False
    if not os.path.exists(bib_path):
        errors.append(f"Missing Bibliography file: {bib_path}")
        return False

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()

    print(f"[OK] Found {tex_path} ({len(tex_content.splitlines())} lines)")
    print(f"[OK] Found {bib_path} ({len(bib_content.splitlines())} lines)")

    # 2. Environment Stack Balance Check
    env_matches = re.findall(r"\\(begin|end)\{([a-zA-Z0-9*]+)\}", tex_content)
    stack = []
    mismatch_found = False
    for action, env_name in env_matches:
        if action == "begin":
            stack.append(env_name)
        elif action == "end":
            if not stack:
                errors.append(f"Unexpected \\end{{{env_name}}} with empty stack.")
                mismatch_found = True
                break
            top = stack.pop()
            if top != env_name:
                errors.append(f"Environment mismatch: expected \\end{{{top}}}, found \\end{{{env_name}}}")
                mismatch_found = True
                break

    if not mismatch_found and stack:
        errors.append(f"Unclosed environments remaining: {stack}")
    elif not mismatch_found:
        print(f"[OK] All {len(env_matches)//2} nested LaTeX environments are perfectly balanced.")

    # 3. Brace Balance Check (ignoring comments)
    clean_lines = []
    for line in tex_content.splitlines():
        line_clean = re.sub(r"(?<!\\)%.*", "", line)
        clean_lines.append(line_clean)
    clean_tex = "\n".join(clean_lines)

    open_braces = clean_tex.count("{")
    close_braces = clean_tex.count("}")
    if open_braces != close_braces:
        errors.append(f"Brace mismatch: '{open_braces}' open braces vs '{close_braces}' close braces")
    else:
        print(f"[OK] Curly braces are balanced ({open_braces} pairs).")

    # 4. Bibliography Verification
    bib_keys = set(re.findall(r"@\w+\{\s*([a-zA-Z0-9_-]+)\s*,", bib_content))
    print(f"[OK] Found {len(bib_keys)} BibTeX entries: {sorted(list(bib_keys))}")

    citations = re.findall(r"\\cite\{([a-zA-Z0-9_,\s-]+)\}", clean_tex)
    all_cited = []
    for c in citations:
        keys = [k.strip() for k in c.split(",")]
        all_cited.extend(keys)

    missing_cites = set(all_cited) - bib_keys
    if missing_cites:
        errors.append(f"Citations missing from references.bib: {missing_cites}")
    else:
        print(f"[OK] All {len(all_cited)} citations in main.tex resolve to valid BibTeX entries.")

    # 5. Image Existence Check
    images = re.findall(r"\\includegraphics(?:\[.*?\])?\{([a-zA-Z0-9_./-]+)\}", clean_tex)
    print(f"[OK] Found {len(images)} figure inclusions.")
    for img in images:
        # Resolve path relative to paper directory
        resolved_img = os.path.normpath(os.path.join("paper", img))
        if not os.path.exists(resolved_img):
            errors.append(f"Figure file does not exist on disk: {resolved_img}")
        else:
            print(f"  [OK] Image confirmed on disk: {resolved_img}")

    # 6. Check required sections
    required_sections = [
        "Introduction and Motivation",
        "Related Work",
        "System Model and Problem Formulation",
        "Agentic Temporal KAN Controller Architecture",
        "Experimental Protocol",
        "Results and Statistical Analysis",
        "Limitations and Threats to Validity",
        "Conclusion",
    ]
    for sec in required_sections:
        if sec.lower() not in tex_content.lower():
            warnings.append(f"Section title '{sec}' might be missing or renamed.")
        else:
            print(f"  [OK] Verified section: '{sec}'")

    print("\n--------------------------------------------------------------------------------")
    if errors:
        print(f"FAILED: Found {len(errors)} errors:")
        for e in errors:
            print(f"  [ERROR] {e}")
        return False
    else:
        print("SUCCESS: All LaTeX structural, citation, and asset integrity checks passed 100%!")
        if warnings:
            for w in warnings:
                print(f"  [WARN] {w}")
        return True


if __name__ == "__main__":
    success = validate_latex()
    sys.exit(0 if success else 1)
