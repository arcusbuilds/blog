"""HashAttention part 1: why the CUDA kernel wins (passes over memory).

Render (run from C:\\Users\\Srijan.Keshri\\dev\\blog, two separate invocations):
  .anim-venv\\Scripts\\manim.exe -qh --fps 30 -o part1_passes_blog   animations\\part1_passes.py PassesBlog
  .anim-venv\\Scripts\\manim.exe -qh --fps 30 -o part1_passes_square animations\\part1_passes.py PassesSquare
Use -ql / -qm to iterate. PassesSquare sets pixel_width = pixel_height itself,
so -ql=480x480, -qm=720x720, -qh=1080x1080. Blog is 16:9 (-qh = 1920x1080).
"""
from manim import *

BG, FG, MUTED, TRACK = "#0e1116", "#e8ecf3", "#8b95a7", "#262d3a"
SLOW, FAST = "#f5a524", "#2dd4a0"  # amber = many passes, green = one pass
FONT = "Segoe UI"

config.background_color = BG


def T(s, size, color=FG, weight=NORMAL):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight)


class PassesBlog(Scene):
    SQ = False
    # ponytail: all timings/sizes in one dict per layout, no per-scene config class
    L = dict(
        head=56, sub=30, cdy=1.5, chip_y=-1.9, bar_w=11.5, bar_h=1.1, bar_y=0.5, head_y=2.8, cap=28,
        count=64, chip=26, foot=20,
        t_a=4.0, t_b=3.0, t_lut=3.5, t_cuda=1.5, hold=1.2, t_end=4.5,
    )

    def __init__(self, *a, **k):
        if self.SQ:
            config.pixel_width = config.pixel_height
            config.frame_width = config.frame_height
        super().__init__(*a, **k)

    def construct(self):
        L, sq = self.L, self.SQ
        bw = L["bar_w"]
        left = -bw / 2

        # --- static pieces
        bar = RoundedRectangle(corner_radius=0.12, width=bw, height=L["bar_h"],
                               fill_color=TRACK, fill_opacity=1, stroke_color=MUTED, stroke_width=3)
        bar.move_to([0, L["bar_y"], 0])
        cap = T("268 MB of int32 codes in GPU memory", L["cap"], MUTED).next_to(bar, UP, buff=0.25)
        foot = T("Animation speed not to scale", L["foot"], MUTED).to_edge(DOWN, buff=0.25)
        head = T("Scoring 67M keys", L["head"], FG, BOLD).move_to([0, L["head_y"], 0])
        sub = T("32 heads × 2M keys, one score per key", L["sub"], MUTED).next_to(head, DOWN, buff=0.2)

        def set_head(old, text, color):
            new = T(text, L["head"], color, BOLD).move_to([0, L["head_y"], 0])
            self.play(FadeOut(old, shift=UP * 0.2), FadeIn(new, shift=UP * 0.2), run_time=0.6)
            return new

        def chip(i, name, ms, passes, color):
            if sq:  # stacked rows under the counter
                g = VGroup(T(name, L["chip"], FG, BOLD), T(ms, L["chip"], color, BOLD),
                           T(passes, L["chip"], MUTED)).arrange(RIGHT, buff=0.35)
                g.move_to([0, L["chip_y"] - 0.65 * i, 0])
            else:  # three columns
                g = VGroup(T(name, L["chip"], FG, BOLD), T(ms, L["chip"] + 8, color, BOLD),
                           T(passes, L["chip"], MUTED)).arrange(DOWN, buff=0.12)
                g.move_to([-4.6 + 4.6 * i, -2.5, 0])
            return g

        def sweep(color, dur, n_from, n_to, label, rate=linear, wf=0.12):
            """Highlight sweeps left->right; each integer of `p` is one full pass."""
            p, hw, ho = ValueTracker(n_from), ValueTracker(wf), ValueTracker(1)

            def frac():
                v = p.get_value()
                f = v % 1
                return 1.0 if (f < 1e-6 and v > 1e-6) else f

            def mk_hl():
                w = hw.get_value() * bw
                r = Rectangle(width=w, height=L["bar_h"] - 0.12, stroke_width=0,
                              fill_color=color, fill_opacity=ho.get_value())
                return r.move_to([left + w / 2 + frac() * (bw - w), L["bar_y"], 0])

            hl = always_redraw(mk_hl)
            cnt = always_redraw(lambda: T(label(int(p.get_value() + 1e-6)), L["count"], color, BOLD)
                                .move_to([0, L["bar_y"] - L["cdy"], 0]))
            return p, hw, ho, hl, cnt

        counter_pos = [0, L["bar_y"] - L["cdy"], 0]

        def final_counter(text, color, size=None):
            return T(text, size or L["count"] - 8, color, BOLD).move_to(counter_pos)

        # --- intro
        self.add(foot)
        self.play(FadeIn(head, shift=DOWN * 0.2), FadeIn(sub, shift=DOWN * 0.2), run_time=0.8)
        self.play(GrowFromEdge(bar, LEFT), FadeIn(cap), run_time=1.0)
        self.wait(0.8 if not sq else 0.4)

        # --- act 1: PyTorch loop, ~100 passes
        head = set_head(VGroup(head, sub), "PyTorch loop: 289 ms", SLOW)
        p, hw, ho, hl, cnt = sweep(SLOW, L["t_a"], 0, 10, lambda n: f"{n} passes" if n != 1 else "1 pass")
        self.add(hl, cnt)
        self.play(p.animate.set_value(10), run_time=L["t_a"], rate_func=linear)
        # too fast to see individual sweeps: widen into a steady glow while the count races on
        self.play(p.animate.set_value(100), hw.animate.set_value(1.0), ho.animate.set_value(0.45),
                  run_time=L["t_b"], rate_func=smooth)
        hl.clear_updaters(); cnt.clear_updaters()
        done = final_counter("~100 passes over memory", SLOW, L["count"] - 14)
        self.play(FadeOut(hl), ReplacementTransform(cnt, done), run_time=0.6)
        c1 = chip(0, "PyTorch loop", "289 ms", "~100 passes", SLOW)
        self.play(FadeIn(c1, shift=UP * 0.2), run_time=0.5)
        self.wait(L["hold"])

        # --- act 2: lookup table, ~15 passes
        head = set_head(head, "Lookup table: 61 ms", SLOW)
        p, hw, ho, hl, cnt = sweep(SLOW, L["t_lut"], 0, 15, lambda n: f"{n} passes" if n != 1 else "1 pass")
        self.play(FadeOut(done), run_time=0.3)
        self.add(hl, cnt)
        self.play(p.animate.set_value(15), run_time=L["t_lut"], rate_func=linear)
        hl.clear_updaters(); cnt.clear_updaters()
        done = final_counter("~15 passes over memory", SLOW, L["count"] - 14)
        self.play(FadeOut(hl), ReplacementTransform(cnt, done), run_time=0.5)
        c2 = chip(1, "Lookup table", "61 ms", "~15 passes", SLOW)
        self.play(FadeIn(c2, shift=UP * 0.2), run_time=0.5)
        self.wait(L["hold"])

        # --- act 3: CUDA kernel, 1 pass
        head = set_head(head, "CUDA kernel: 2.46 ms", FAST)
        sub2 = T("one __popc per key", L["sub"], MUTED).next_to(head, DOWN, buff=0.2)
        p, hw, ho, hl, cnt = sweep(FAST, L["t_cuda"], 0, 1, lambda n: f"{n} pass" if n == 1 else "0 passes")
        self.play(FadeOut(done), FadeIn(sub2), run_time=0.3)
        self.add(hl, cnt)
        self.play(p.animate.set_value(1), run_time=L["t_cuda"], rate_func=smooth)
        hl.clear_updaters(); cnt.clear_updaters()
        done = final_counter("1 pass over memory", FAST, L["count"] - 14)
        self.play(FadeOut(hl), ReplacementTransform(cnt, done), run_time=0.5)
        c3 = chip(2, "CUDA kernel", "2.46 ms", "1 pass", FAST)
        self.play(FadeIn(c3, shift=UP * 0.2), run_time=0.5)
        self.wait(L["hold"])

        # --- end card
        everything = VGroup(head, sub2, bar, cap, done, c1, c2, c3, foot)
        big = T("117x", 150 if not sq else 160, FAST, BOLD)
        l2 = T("faster on the same GPU.", 52 if not sq else 44, FG, BOLD)
        l3 = T("Same math, fewer trips to memory.", 34 if not sq else 30, MUTED)
        end = VGroup(big, l2, l3).arrange(DOWN, buff=0.35).move_to(ORIGIN)
        self.play(FadeOut(everything), run_time=0.6)
        self.play(FadeIn(big, scale=0.8), run_time=0.7)
        self.play(FadeIn(l2, shift=UP * 0.2), run_time=0.5)
        self.play(FadeIn(l3, shift=UP * 0.2), run_time=0.5)
        self.wait(L["t_end"] - 2.3)


class PassesSquare(PassesBlog):
    SQ = True
    L = dict(
        head=44, sub=26, cdy=1.3, chip_y=-1.7, bar_w=7.0, bar_h=1.1, bar_y=0.8, head_y=3.0, cap=22,
        count=50, chip=24, foot=18,
        t_a=2.0, t_b=1.5, t_lut=2.0, t_cuda=1.1, hold=0.4, t_end=3.6,
    )
