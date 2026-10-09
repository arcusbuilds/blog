"""HashAttention part 0: the map (why a model rereads its whole KV cache for every token).

Render (run from C:\\Users\\Srijan.Keshri\\dev\\blog, two separate invocations):
  .anim-venv\\Scripts\\manim.exe -qh --fps 30 --media_dir animations\\out\\_media -o part0_cache_blog   animations\\part0_cache.py CacheBlog
  .anim-venv\\Scripts\\manim.exe -qh --fps 30 --media_dir animations\\out\\_media -o part0_cache_square animations\\part0_cache.py CacheSquare
Use -ql / -qm to iterate. CacheSquare sets pixel_width = pixel_height itself,
so -ql=480x480, -qm=720x720, -qh=1080x1080. Blog is 16:9 (-qh = 1920x1080).
All numbers are arithmetic from Llama-3.1-8B shapes (128 KB of K+V per token), not measurements.
"""
from manim import *

BG, FG, MUTED, TRACK = "#0e1116", "#e8ecf3", "#8b95a7", "#262d3a"
SLOW, FAST = "#f5a524", "#2dd4a0"  # amber = slow / reads everything, green = fast / reads little
FONT = "Segoe UI"

config.background_color = BG


def T(s, size, color=FG, weight=NORMAL):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


def size_str(n):  # n tokens * 128 KB
    kb = n * 128
    if kb < 1024:
        return f"{kb} KB"
    if kb < 1024 * 1024:
        return f"{kb / 1024:.0f} MB"
    return f"{kb / 1048576:.0f} GB"


class CacheBlog(Scene):
    SQ = False
    # ponytail: all timings/sizes in one dict per layout, same as part 1
    L = dict(
        head=52, sub=28, cap=28, count=56, foot=20, big=34, key=30,
        n_tok=16, bar_w=11.5, bar_h=1.0, bar_y=0.6, head_y=2.9, cdy=1.55, solo=5,
        t_step=0.55, t_fill=3.0, hold=1.0, t_end=4.0, show_ms=True,
    )

    def __init__(self, *a, **k):
        if self.SQ:
            config.pixel_width = config.pixel_height
            config.frame_width = config.frame_height
        super().__init__(*a, **k)

    def fit(self, m, maxw=None):
        maxw = maxw or config.frame_width - 0.8
        if m.width > maxw:
            m.scale_to_fit_width(maxw)
        return m

    def construct(self):
        L = self.L
        bw, bh, by, n_tok = L["bar_w"], L["bar_h"], L["bar_y"], L["n_tok"]
        gap = 0.08
        cw = (bw - gap * (n_tok - 1)) / n_tok
        left = -bw / 2
        hpos = [0, L["head_y"], 0]
        cpos = [0, by - L["cdy"], 0]

        def cell(i):
            r = RoundedRectangle(corner_radius=0.06, width=cw, height=bh, fill_color=TRACK,
                                 fill_opacity=1, stroke_color=MUTED, stroke_width=2)
            return r.move_to([left + cw / 2 + i * (cw + gap), by, 0])

        cells = [cell(i) for i in range(n_tok)]
        foot = T("Animation speed not to scale", L["foot"], MUTED).to_edge(DOWN, buff=0.25)

        def head_t(text, color):
            return self.fit(T(text, L["head"], color, BOLD).move_to(hpos))

        def set_head(old, text, color):
            new = head_t(text, color)
            self.play(FadeOut(old, shift=UP * 0.2), FadeIn(new, shift=UP * 0.2), run_time=0.6)
            return new

        def right_edge(n):  # right edge of the first n cells
            return left + n * cw + (n - 1) * gap

        def sweep(n, dur, opacity=0.6):
            w = right_edge(n) - left
            hl = Rectangle(width=w, height=bh - 0.1, stroke_width=0, fill_color=SLOW, fill_opacity=opacity)
            hl.move_to([left + w / 2, by, 0]).set_z_index(1)
            self.play(GrowFromEdge(hl, LEFT), run_time=dur, rate_func=linear)
            return hl

        def counter_group(n):
            tok = f"{n} token" + ("" if n == 1 else "s") if n < 1000 else f"{n / 1024:.1f}K tokens"
            return VGroup(T(tok, L["count"], SLOW, BOLD),
                          T(f"× 128 KB = {size_str(n)}", L["count"] - 20, MUTED)
                          ).arrange(DOWN, buff=0.12).move_to(cpos)

        # --- intro
        self.add(foot)
        head = head_t("The KV cache: a model's memory", FG)
        sub = self.fit(T("one entry per token, 128 KB each", L["sub"], MUTED).next_to(head, DOWN, buff=0.2))
        self.play(FadeIn(head, shift=DOWN * 0.2), FadeIn(sub, shift=DOWN * 0.2), run_time=0.8)
        self.wait(0.4)
        head = set_head(VGroup(head, sub), "Every new token rereads all of it", SLOW)

        # --- act 1: tokens arrive one by one, each rereads everything
        n_solo = L["solo"]
        cnt = None
        for i in range(n_solo):
            self.play(FadeIn(cells[i], scale=0.6), run_time=0.25)
            new = counter_group(i + 1)
            if cnt is None:
                self.add(new)
            else:
                self.remove(cnt); self.add(new)
            cnt = new
            hl = sweep(i + 1, L["t_step"])
            self.play(FadeOut(hl), run_time=0.15)

        # fast-forward: fill the row, counter races to 32K, glow replaces individual sweeps
        p = ValueTracker(n_solo)
        self.remove(cnt)
        live = always_redraw(lambda: counter_group(max(1, int(p.get_value()))))
        self.add(live)
        glow = Rectangle(width=bw, height=bh - 0.1, stroke_width=0, fill_color=SLOW, fill_opacity=0.0)
        glow.move_to([0, by, 0]).set_z_index(1)
        self.add(glow)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in cells[n_solo:]], lag_ratio=0.15),
                  p.animate.set_value(32768), glow.animate.set_fill(SLOW, opacity=0.5),
                  run_time=L["t_fill"], rate_func=smooth)
        live.clear_updaters()
        done = VGroup(T("32K tokens = 4 GB", L["count"], SLOW, BOLD),
                      T("read for every new token", L["count"] - 20, FG)
                      ).arrange(DOWN, buff=0.12).move_to(cpos)
        self.play(ReplacementTransform(live, done), run_time=0.6)
        extras = VGroup()
        if L["show_ms"]:
            ms = self.fit(T("~17 ms per token on a 256 GB/s laptop GPU, just moving bytes", L["cap"], SLOW))
            ms.move_to([0, cpos[1] - 1.45, 0])
            note = T("arithmetic, not measured", L["foot"], MUTED).next_to(ms, DOWN, buff=0.12)
            extras = VGroup(ms, note)
            self.play(FadeIn(extras, shift=UP * 0.2), run_time=0.6)
        self.wait(L["hold"] + (1.2 if L["show_ms"] else 0))

        # --- act 2: most keys barely matter
        head = set_head(head, "Most keys barely matter", FAST)
        keep = [2, 7, 12] if n_tok >= 16 else [1, 4, 8]
        ratio = VGroup(T("e.g. 1,000 of 32,000 keys", L["count"] - 14, FAST, BOLD),
                       T("illustrative", L["foot"], MUTED)).arrange(DOWN, buff=0.12).move_to(cpos)
        self.play(FadeOut(glow), FadeOut(extras), ReplacementTransform(done, ratio),
                  *[c.animate.set_opacity(0.25) for i, c in enumerate(cells) if i not in keep],
                  *[c.animate.set_fill(FAST, 1).set_stroke(FAST) for i, c in enumerate(cells) if i in keep],
                  run_time=0.9)
        self.wait(L["hold"] + 0.6)

        # --- act 3: find them cheaply
        head = set_head(head, "Find them cheaply", FAST)
        full_w = bw * 0.72
        full = RoundedRectangle(corner_radius=0.1, width=full_w, height=0.8, fill_color=SLOW,
                                fill_opacity=1, stroke_width=0).move_to([0, by, 0])
        lab1 = T("full key: 256 bytes", L["key"] + 6, SLOW, BOLD).next_to(full, UP, buff=0.3)
        self.play(FadeOut(VGroup(*cells, ratio)), GrowFromCenter(full), FadeIn(lab1), run_time=0.8)
        self.wait(0.6)
        tiny = RoundedRectangle(corner_radius=0.02, width=full_w / 64, height=0.8, fill_color=FAST,
                                fill_opacity=1, stroke_width=0).move_to([0, by, 0])
        lab2 = T("32-bit code: 4 bytes", L["key"] + 6, FAST, BOLD).next_to(tiny, UP, buff=0.3)
        less = T("64x less to read", L["count"], FAST, BOLD).move_to(cpos)
        self.play(Transform(full, tiny), ReplacementTransform(lab1, lab2), run_time=1.1, rate_func=smooth)
        self.play(FadeIn(less, shift=UP * 0.2), run_time=0.5)
        self.wait(L["hold"] + 1.0)

        # --- end card (all text high contrast)
        title = self.fit(T("HashAttention from scratch", L["head"] + 12, FG, BOLD), config.frame_width - 0.6)
        subt = self.fit(T("4 CUDA kernels. Part 0: the map.", L["big"] + 4, FAST, BOLD), config.frame_width - 0.6)
        end = VGroup(title, subt).arrange(DOWN, buff=0.45).move_to(ORIGIN)
        self.play(FadeOut(VGroup(head, full, lab2, less, foot)), run_time=0.6)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.play(FadeIn(subt, shift=UP * 0.2), run_time=0.5)
        self.wait(L["t_end"] - 1.7)


class CacheSquare(CacheBlog):
    SQ = True
    L = dict(
        head=40, sub=28, cap=24, count=56, foot=20, big=34, key=32,
        n_tok=10, bar_w=7.0, bar_h=1.0, bar_y=0.8, head_y=3.0, cdy=1.5, solo=3,
        t_step=0.4, t_fill=1.8, hold=0.4, t_end=3.2, show_ms=False,
    )
