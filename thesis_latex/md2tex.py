#!/usr/bin/env python3
"""Turn the thesis drafts in docs/thesis/ (Markdown) into LaTeX chapters for the ctuthesis class.

    python3 thesis_latex/md2tex.py            # writes thesis_latex/chapters/*.tex and thesis_latex/abstract-english.tex

What it converts:
- headings: "Chapter N. Title" and "N.M Title" become \\chapter and \\section with labels ch:N and sec:N.M;
  "Appendix X. Title" becomes an appendix chapter with label app:X;
- paragraphs, bold and italic, `code`, lists (nested by indentation), pipe tables (booktabs; tabularx where the
  text is long, xltabular where the table is long), fenced code (fvextra's Verbatim with line breaking);
- images "![**Figure N.M.** caption](../figures/name.png)" become figures with label fig:name;
- citations "Name et al. (2021)", "(Name, 2021; Other, 2020)", "Name (2019, pp. 1 and 43)" become natbib commands,
  through the table CITES below;
- references "Section 6.2", "Chapter 3", "Figure 2.1", "Appendix E" become \\ref.
Reference lists in the drafts are dropped: the bibliography comes from references.bib.

Unicode symbols (±, →, ≈, σ, superscript and subscript digits, ...) are left in the text and declared in main.tex,
so they work in verbatim text too. After a run, check the output by hand: the converter is meant to save typing,
not to be right about every corner of Markdown.
"""
import os, re, sys

SRC = "docs/thesis"
OUT = "thesis_latex"
CHAPTERS = ["abstract_introduction", "ch2_background_related_work", "ch3_pricing_proofs", "ch4_task_data_models",
            "ch5_solver_results", "ch6_symmetry_buys_costs", "ch7_sum_midpoint", "ch8_discussion_conclusion"]

CITES = {
    ("Abbe et al.", "2023"): "abbe2023generalization", ("Baez", "2009"): "baez2009torsors",
    ("Barrett et al.", "2018"): "barrett2018measuring", ("Battaglia et al.", "2018"): "battaglia2018relational",
    ("Bietti et al.", "2021"): "bietti2021sample", ("Boix-Adserà et al.", "2024"): "boixadsera2024when",
    ("Bronstein et al.", "2021"): "bronstein2021geometric", ("Camposampiero et al.", "2024"): "camposampiero2024towards",
    ("Carpenter et al.", "1990"): "carpenter1990what", ("Chollet", "2019"): "chollet2019measure",
    ("Cohen and Welling", "2016"): "cohen2016group", ("Elesedy and Zaidi", "2021"): "elesedy2021provably",
    ("Finzi et al.", "2021"): "finzi2021residual", ("Gallier", "2011"): "gallier2011geometric",
    ("Geirhos et al.", "2020"): "geirhos2020shortcut", ("Hersche et al.", "2023"): "hersche2023neuro",
    ("Hersche et al.", "2024"): "hersche2024towards", ("Hu et al.", "2021"): "hu2021stratified",
    ("Kondor and Trivedi", "2018"): "kondor2018generalization", ("Lake et al.", "2017"): "lake2017building",
    ("Liu et al.", "2023"): "liu2023omnigrok", ("Marcus", "2018"): "marcus2018deep",
    ("Małkiński and Mańdziuk", "2025a"): "malkinski2025airaven", ("Małkiński and Mańdziuk", "2025b"): "malkinski2025deep",
    ("Mitchell", "1980"): "mitchell1980need", ("Mohamadi et al.", "2024"): "mohamadi2024why",
    ("Mondal et al.", "2023"): "mondal2023learning", ("Nanda et al.", "2023"): "nanda2023progress",
    ("Petrache and Trivedi", "2023"): "petrache2023approximation", ("Power et al.", "2022"): "power2022grokking",
    ("Ravanbakhsh et al.", "2017"): "ravanbakhsh2017equivariance", ("Raven", "1938"): "raven1938progressive",
    ("Reem", "2017"): "reem2017remarks", ("Sannai et al.", "2021"): "sannai2021improved",
    ("Sokolić et al.", "2017"): "sokolic2017generalization", ("Spratley et al.", "2020"): "spratley2020closer",
    ("Stevens", "1946"): "stevens1946theory", ("Sutton", "2019"): "sutton2019bitter",
    ("Tahmasebi and Jegelka", "2023"): "tahmasebi2023exact", ("Trask et al.", "2018"): "trask2018neural",
    ("Varma et al.", "2023"): "varma2023explaining", ("Wang et al.", "2020"): "wang2020abstract",
    ("Wang et al.", "2022"): "wang2022approximately", ("Wang et al.", "2023a"): "wang2023general",
    ("Wang et al.", "2023b"): "wang2023surprising", ("Wolpert", "1996"): "wolpert1996lack",
    ("Wu et al.", "2020"): "wu2020scattering", ("Xu et al.", "2020"): "xu2020what", ("Xu et al.", "2021"): "xu2021how",
    ("Zaheer et al.", "2017"): "zaheer2017deep", ("Zhang et al.", "2019a"): "zhang2019raven",
    ("Zhang et al.", "2019b"): "zhang2019learning", ("Zhang et al.", "2021"): "zhang2021abstract",
    ("Zhang et al.", "2022"): "zhang2022learning", ("Zhou et al.", "2020"): "zhou2020graph",
    ("Zhuo and Kankanhalli", "2020"): "zhuo2020solving", ("Šourek et al.", "2018"): "sourek2018lifted",
    ("Šourek et al.", "2021"): "sourek2021beyond",
}
NAME = r"[A-ZŠŁ][\w'\-]+(?: (?:and|&) [A-ZŠŁ][\w\-]+| et al\.)?"
EXTRA = r"(?:Section|Sections|Theorem|Lemma|Table|Proposition|Chapter|Definition|p\.|pp\.)[^;)]*"
POST = {   # exact replacements in the LaTeX output, per chapter file: the places the converter cannot know
    "ch3.tex": [
        (r"We write it val\_F(a).", r"We write it \(\mathrm{val}_F(a)\)."),
        ("\nval\\_σ(F)(σ(a)) = val\\_F(a).\n", "\n\\[ \\mathrm{val}_{\\sigma(F)}(\\sigma(a)) = \\mathrm{val}_F(a). \\]\n"),
    ],
}
FIGS = {}            # "2.1" -> "iraven_puzzle", filled from the image lines before conversion
missing = set()


# ------------------------------------------------------------------------------------------------- inline text
class Protect:
    """Placeholders for LaTeX produced before escaping, so escaping leaves it alone."""

    def __init__(self):
        self.items = []

    def __call__(self, latex):
        self.items.append(latex)
        return f"\x00{len(self.items) - 1}\x00"

    def restore(self, s):
        while "\x00" in s:
            s = re.sub("\x00(\\d+)\x00", lambda m: self.items[int(m.group(1))], s)
        return s


def esc(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def cite_key(name, year):
    k = CITES.get((name, year))
    if k is None:
        missing.add((name, year))
    return k


def refs(s, P):
    def secs(m):
        nums = re.findall(r"\d+(?:\.\d+)+", m.group(2))
        out = m.group(2)
        for n in sorted(set(nums), key=len, reverse=True):
            out = re.sub(rf"(?<![\d.]){re.escape(n)}(?![\d.]*\d)", P(rf"\ref{{sec:{n}}}"), out)
        return m.group(1) + P("~") + out
    s = re.sub(r"\b(Sections?) (\d+(?:\.\d+)+(?:(?:, | and | or | to |–)\d+(?:\.\d+)+)*)", secs, s)

    def chaps(m):
        out = re.sub(r"\d+", lambda n: P(rf"\ref{{ch:{n.group(0)}}}"), m.group(2))
        return m.group(1) + P("~") + out
    s = re.sub(r"\b(Chapters?) (\d+(?:(?:, | and | to |–)\d+)*)(?![\d.]*\d)", chaps, s)

    def figs(m):
        nums = re.findall(r"\d+\.\d+", m.group(2))
        out = m.group(2)
        for n in nums:
            out = out.replace(n, P(rf"\ref{{fig:{FIGS.get(n, n)}}}"), 1)
        return m.group(1) + P("~") + out
    s = re.sub(r"\b(Figures?) (\d+\.\d+(?: and \d+\.\d+)?)", figs, s)
    s = re.sub(r"\bAppendix ([A-F])\.(\d+)\b", lambda m: P("Appendix~") + P(rf"\ref{{sec:{m.group(1)}.{m.group(2)}}}"), s)
    s = re.sub(r"\bAppendix ([A-F])\b", lambda m: P("Appendix~") + P(rf"\ref{{app:{m.group(1)}}}"), s)
    return s


def cites(s, P):
    def narrative(m):
        k = cite_key(m.group(1), m.group(2))
        if not k:
            return m.group(0)
        extra = m.group(3)[2:] if m.group(3) else ""
        return P(rf"\citet[{esc(extra)}]{{{k}}}" if extra else rf"\citet{{{k}}}")
    s = re.sub(rf"({NAME}) \((\d{{4}}[a-z]?)((?:, {EXTRA})?)\)", narrative, s)

    def inline_(m):
        k = cite_key(m.group(1), m.group(2))
        if not k:
            return m.group(0)
        extra = m.group(3)[2:] if m.group(3) else ""
        return P(rf"\citealp[{esc(extra)}]{{{k}}}" if extra else rf"\citealp{{{k}}}")
    s = re.sub(rf"({NAME}),? (\d{{4}}[a-z]?)\b((?:, {EXTRA})?)", inline_, s)
    return s


def inline(s):
    P = Protect()
    s = re.sub(r"`([^`]+)`", lambda m: P(r"\texttt{" + esc(m.group(1)) + "}"), s)
    s = re.sub(r"\$([^$\n]+?)\$", lambda m: P(r"\(" + m.group(1) + r"\)"), s)       # inline LaTeX math, as written
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", lambda m: m.group(1) + " " + P(r"\url{" + m.group(2) + "}"), s)
    s = cites(s, P)
    s = refs(s, P)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r'(^|(?<=[\s(\[{\x00–/]))"', "``", s)
    s = s.replace('"', "''")
    s = s.replace("§", r"\S{}")
    return P.restore(s)


# --------------------------------------------------------------------------------------------------- blocks
def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    width = [max(len(r[j]) for r in rows) for j in range(n)]
    cells = [[inline(c) for c in r] for r in rows]
    body = [" & ".join(r) + r" \\" for r in cells]
    head, rest = body[0], body[1:]
    if sum(width) <= 80:
        spec = "@{}" + "l" * n + "@{}"
        return ["", r"\begin{center}\small", rf"\begin{{tabular}}{{{spec}}}", r"\toprule", head, r"\midrule"] + rest + \
               [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    spec = "@{}" + "".join("Y" if w > 18 else "l" for w in width) + "@{}"
    if "Y" not in spec:
        spec = "@{}" + "Y" * n + "@{}"
    if len(rows) > 22:
        return ["", r"{\small", rf"\begin{{xltabular}}{{\linewidth}}{{{spec}}}", r"\toprule", head, r"\midrule",
                r"\endhead"] + rest + [r"\bottomrule", r"\end{xltabular}}", ""]
    return ["", r"\begin{center}\small", rf"\begin{{tabularx}}{{\linewidth}}{{{spec}}}", r"\toprule", head,
            r"\midrule"] + rest + [r"\bottomrule", r"\end{tabularx}", r"\end{center}", ""]


def listing(lines):
    """Nested itemize/enumerate from list lines; continuation lines are joined to their item."""
    items = []                                    # (indent, kind, text)
    for l in lines:
        m = re.match(r"^(\s*)(-|\d+\.) (.*)$", l)
        if m:
            items.append([len(m.group(1)), "enumerate" if m.group(2)[0].isdigit() else "itemize", m.group(3)])
        else:
            items[-1][2] += " " + l.strip()
    out, stack = [], []
    for ind, kind, text in items:
        while stack and ind < stack[-1][0]:
            out.append(rf"\end{{{stack.pop()[1]}}}")
        if not stack or ind > stack[-1][0]:
            stack.append((ind, kind))
            out.append(rf"\begin{{{kind}}}")
        out.append(r"\item " + inline(text))
    while stack:
        out.append(rf"\end{{{stack.pop()[1]}}}")
    return [""] + out + [""]


def figure(line):
    m = re.match(r"!\[\*\*Figure (\d+\.\d+)\.\*\* (.*)\]\(\.\./figures/([\w\-]+)\.png\)\s*$", line)
    num, cap, name = m.groups()
    return ["", r"\begin{figure}[tbp]", r"\centering", rf"\includegraphics[width=\linewidth]{{figures/{name}}}",
            rf"\caption{{{inline(cap)}}}\label{{fig:{name}}}", r"\end{figure}", ""]


def heading(line, appendix):
    level = len(line) - len(line.lstrip("#"))
    text = re.sub(r"\s*\(draft, [\d-]+\)\s*$", "", line[level:].strip())
    m = re.match(r"Chapter (\d+)\. (.*)", text)
    if m:
        return [rf"\chapter{{{inline(m.group(2))}}}\label{{ch:{m.group(1)}}}", ""]
    m = re.match(r"Appendix ([A-F])\. (.*)", text)
    if m:
        return [rf"\chapter{{{inline(m.group(2))}}}\label{{app:{m.group(1)}}}", ""]
    m = re.match(r"([A-F]|\d+)((?:\.\d+)+) (.*)", text)
    if m:
        num = m.group(1) + m.group(2)
        cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[num.count(".")]
        return [rf"\{cmd}{{{inline(m.group(3))}}}\label{{sec:{num}}}", ""]
    return [rf"\section*{{{inline(text)}}}", ""]


REFLINE = re.compile(r"^- [A-ZŠŁ][\w\-' ]+, [A-Z][\w.\- ]*[.,].*\(\d{4}[a-z]?\)")


def convert(md, appendix=False):
    lines = md.split("\n")
    out, i, skip = [], 0, False
    while i < len(lines):
        l = lines[i]
        if l.startswith("#"):
            skip = bool(re.match(r"#+ (References|Appendices)", l))
            if not skip:
                out += heading(l, appendix)
            i += 1
            continue
        if skip:
            i += 1
            continue
        if l.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            if l.strip() == "```math":                  # LaTeX math, passed through as written (GitHub renders it too)
                out += ["", r"\begin{align*}"] + lines[i + 1:j] + [r"\end{align*}", ""]
                i = j + 1
                continue
            out += ["", r"\begin{Verbatim}[fontsize=\small,breaklines]"] + lines[i + 1:j] + [r"\end{Verbatim}", ""]
            i = j + 1
            continue
        if l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out += table(lines[i:j])
            i = j
            continue
        if l.startswith("!["):
            out += figure(l)
            i += 1
            continue
        if re.match(r"^\s*(-|\d+\.) ", l):
            j = i
            while j < len(lines) and lines[j].strip() and (re.match(r"^\s*(-|\d+\.) ", lines[j]) or
                                                          lines[j].startswith(" ")):
                j += 1
            block = [x for x in lines[i:j]]
            if all(REFLINE.match(x) or x.startswith(" ") for x in block):
                i = j
                continue
            out += listing(block)
            i = j
            continue
        if l.strip() in ("", "---"):
            out.append("")
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|```|\||!\[|\s*(-|\d+\.) |---$)", lines[j]):
            j += 1
        out.append(inline(" ".join(x.strip() for x in lines[i:j])))
        out.append("")
        i = j
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main():
    os.makedirs(os.path.join(OUT, "chapters"), exist_ok=True)
    srcs = {n: open(os.path.join(SRC, n + ".md"), encoding="utf-8").read() for n in CHAPTERS + ["appendices"]}
    for s in srcs.values():
        for num, name in re.findall(r"!\[\*\*Figure (\d+\.\d+)\.\*\*.*?\]\(\.\./figures/([\w\-]+)\.png\)", s):
            FIGS[num] = name
    intro = srcs["abstract_introduction"]
    a = intro.index("## Abstract")
    b = intro.index("## Chapter 1.")
    abstract = intro[a:b].split("\n", 1)[1]
    kw = re.search(r"\*\*Keywords:\*\*\s*(.*?)(?:\n\s*\n|$)", abstract, flags=re.S)
    abstract = abstract[:kw.start()].strip() if kw else abstract
    open(os.path.join(OUT, "abstract-english.tex"), "w", encoding="utf-8").write(convert(abstract))
    if kw:
        open(os.path.join(OUT, "keywords-english.txt"), "w", encoding="utf-8").write(" ".join(kw.group(1).split()) + "\n")
    srcs["abstract_introduction"] = intro[b:].replace("## Chapter 1.", "# Chapter 1.", 1).replace("\n### ", "\n## ")
    for k, n in enumerate(CHAPTERS, 1):
        tex = convert(srcs[n])
        for old, new in POST.get(f"ch{k}.tex", []):
            if old not in tex:
                print(f"ch{k}.tex: hand fix not applied, text not found: {old[:50]!r}")
            tex = tex.replace(old, new)
        open(os.path.join(OUT, "chapters", f"ch{k}.tex"), "w", encoding="utf-8").write(tex)
    open(os.path.join(OUT, "chapters", "appendices.tex"), "w", encoding="utf-8").write(convert(srcs["appendices"], True))
    if missing:
        print("citations not in CITES (left as text):", sorted(missing))
    print("figures:", FIGS)


if __name__ == "__main__":
    main()
