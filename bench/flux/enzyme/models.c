// flux's benchmark models in C, differentiated by Enzyme (LLVM-level reverse mode), on the inputs
// `examples/flux_programs.rs` writes and checked against flux's interpreter.
//
//     clang -O3 -march=native -fplugin=ClangEnzyme-21.so models.c -lm -o models && ./models DIR
//
// Prints one Markdown row per model: median run time and the largest difference from flux's outputs
// relative to the largest output. The FFT is a plain iterative radix-2 transform (Enzyme
// differentiates the code it is given; a library FFT would need hand-written derivative rules), the
// dense layer a loop nest the compiler vectorizes.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FS 48000.0f
#define PI_F 3.14159265358979323846f

extern int enzyme_dup, enzyme_const, enzyme_dupnoneed;
void __enzyme_autodiff(void *, ...);

// INPUT ===========================================================================================

typedef struct {
    float *data;
    size_t len;
} Array;

static Array load(const char *dir, const char *case_name, const char *which) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s_%s.npy", dir, case_name, which);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    unsigned char head[10];
    fseek(f, 0, SEEK_SET);
    if (fread(head, 1, 10, f) != 10) exit(1);
    size_t header = head[8] | (head[9] << 8);
    Array a = {0};
    a.len = (size - 10 - header) / 4;
    a.data = malloc(a.len * 4);
    fseek(f, 10 + header, SEEK_SET);
    if (fread(a.data, 4, a.len, f) != a.len) exit(1);
    fclose(f);
    return a;
}

static double error(const float *got, const Array want) {
    double top = 1e-30, worst = 0.0;
    for (size_t i = 0; i < want.len; i++) top = fmax(top, fabs(want.data[i]));
    for (size_t i = 0; i < want.len; i++) worst = fmax(worst, fabs((double)got[i] - want.data[i]));
    return worst / top;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int by_value(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// The median of runs filling about a second (at least 3, at most 200).
#define MEDIAN_TIME(result, body)                                                   \
    do {                                                                            \
        body;                                                                       \
        double times[200];                                                          \
        int count = 0;                                                              \
        double start = now();                                                       \
        while (count < 3 || (now() - start < 1.0 && count < 200)) {                 \
            double t = now();                                                       \
            body;                                                                   \
            times[count++] = now() - t;                                             \
        }                                                                           \
        qsort(times, count, sizeof(double), by_value);                              \
        result = times[count / 2];                                                  \
    } while (0)

static void row(const char *model, double seconds, double err) {
    if (seconds >= 1.0)
        printf("| %s | Enzyme (C, clang -O3) | %.2f s | %.1e |\n", model, seconds, err);
    else if (seconds >= 1e-3)
        printf("| %s | Enzyme (C, clang -O3) | %.2f ms | %.1e |\n", model, seconds * 1e3, err);
    else
        printf("| %s | Enzyme (C, clang -O3) | %.1f µs | %.1e |\n", model, seconds * 1e6, err);
    fflush(stdout);
}

// THE MODELS ======================================================================================

static void one_pole(float cutoff, const float *xs, float s0, float *ys, int n) {
    float a = 1.0f - expf(-2.0f * PI_F * cutoff / FS);
    float s = s0;
    for (int i = 0; i < n; i++) {
        s = s + a * (xs[i] - s);
        ys[i] = s;
    }
}

static float one_pole_loss(const float *cutoff, const float *xs, const float *targets, const float *s0, int n) {
    float a = 1.0f - expf(-2.0f * PI_F * cutoff[0] / FS);
    float s = s0[0], total = 0.0f;
    for (int i = 0; i < n; i++) {
        s = s + a * (xs[i] - s);
        float e = s - targets[i];
        total += e * e;
    }
    return total / n;
}

// In-place radix-2 FFT of length n (a power of two); `tw` holds cos / sin of 2πk/n for k < n/2.
// inverse: conjugate twiddles, no scaling.
static void fft(float *re, float *im, int n, const float *tw_cos, const float *tw_sin, int inverse) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (int start = 0; start < n; start += len) {
            for (int k = 0; k < half; k++) {
                float wr = tw_cos[k * step], wi = inverse ? tw_sin[k * step] : -tw_sin[k * step];
                int p = start + k, q = p + half;
                float xr = re[q] * wr - im[q] * wi, xi = re[q] * wi + im[q] * wr;
                re[q] = re[p] - xr; im[q] = im[p] - xi;
                re[p] += xr; im[p] += xi;
            }
        }
    }
}

static void twiddles(int n, float **c, float **s) {
    *c = malloc(sizeof(float) * n / 2);
    *s = malloc(sizeof(float) * n / 2);
    for (int k = 0; k < n / 2; k++) {
        (*c)[k] = (float)cos(2.0 * M_PI * k / n);
        (*s)[k] = (float)sin(2.0 * M_PI * k / n);
    }
}

// rfft -> per-bin gain -> irfft -> dense -> tanh; the mean energy.
static float spectral_loss(const float *x, const float *gain, const float *w, const float *tw_cos, const float *tw_sin, int batch, int len, int width) {
    float *re = malloc(sizeof(float) * len), *im = malloc(sizeof(float) * len);
    float *h = malloc(sizeof(float) * width);
    float total = 0.0f;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < len; i++) {
            re[i] = x[b * len + i];
            im[i] = 0.0f;
        }
        fft(re, im, len, tw_cos, tw_sin, 0);
        // the gained half spectrum, mirrored (irfft's Hermitian extension)
        for (int k = 0; k <= len / 2; k++) {
            re[k] *= gain[k];
            im[k] *= gain[k];
        }
        im[0] = 0.0f;
        im[len / 2] = 0.0f;
        for (int k = 1; k < len / 2; k++) {
            re[len - k] = re[k];
            im[len - k] = -im[k];
        }
        fft(re, im, len, tw_cos, tw_sin, 1);
        for (int j = 0; j < width; j++) h[j] = 0.0f;
        for (int i = 0; i < len; i++) {
            float y = re[i] / len;
            for (int j = 0; j < width; j++) h[j] += y * w[i * width + j];
        }
        for (int j = 0; j < width; j++) {
            float t = tanhf(h[j]);
            total += t * t;
        }
    }
    free(re);
    free(im);
    free(h);
    return total / (batch * width);
}

static const int RES[3][3] = {{512, 128, 512}, {128, 32, 128}, {32, 8, 32}};

// |STFT| of x (zero-padded by window / 2 on the left, enough on the right to land on a hop), Hann
// window, n_fft = window: frames x (window / 2 + 1) magnitudes into `mag`; returns the frame count.
static int stft_magnitude(const float *x, int n, int hop, int window, const float *tw_cos, const float *tw_sin, float *mag) {
    int low = window / 2, high = window / 2;
    while ((n + low + high - window) % hop) high++;
    int count = 1 + (n + low + high - window) / hop, bins = window / 2 + 1;
    float *re = malloc(sizeof(float) * window), *im = malloc(sizeof(float) * window);
    for (int f = 0; f < count; f++) {
        for (int i = 0; i < window; i++) {
            int at = f * hop + i - low;
            float hann = 0.5f - 0.5f * cosf(2.0f * PI_F * i / window);
            re[i] = (at >= 0 && at < n) ? x[at] * hann : 0.0f;
            im[i] = 0.0f;
        }
        fft(re, im, window, tw_cos, tw_sin, 0);
        for (int k = 0; k < bins; k++) mag[f * bins + k] = sqrtf(fmaxf(re[k] * re[k] + im[k] * im[k], 1e-8f));
    }
    free(re);
    free(im);
    return count;
}

typedef struct {
    const float *cos[3], *sin[3];
} Tables;

// peaking EQ (log frequency, gain / 10 dB, Q 1) into tanh drive; the multi-resolution STFT loss.
static float eq_drive_loss(const float *log_freq, const float *g10, const float *drive, const float *s0a, const float *s0b, const float *xs, const float *target, const Tables *tables, int n) {
    float f = expf(log_freq[0]), gain_db = g10[0] * 10.0f;
    float wv = 2.0f * PI_F * f / FS, cos_w = cosf(wv), alpha = sinf(wv) / 2.0f;
    float a = sqrtf(powf(10.0f, gain_db / 20.0f)), a0 = 1.0f + alpha / a;
    float b0 = (1.0f + alpha * a) / a0, b1 = -2.0f * cos_w / a0, b2 = (1.0f - alpha * a) / a0;
    float a1 = -2.0f * cos_w / a0, a2 = (1.0f - alpha / a) / a0;
    float *ys = malloc(sizeof(float) * n);
    float sa = s0a[0], sb = s0b[0];
    for (int i = 0; i < n; i++) {
        float x = xs[i], y = b0 * x + sa;
        sa = b1 * x - a1 * y + sb;
        sb = b2 * x - a2 * y;
        ys[i] = tanhf(y * drive[0]);
    }
    float total = 0.0f;
    for (int r = 0; r < 3; r++) {
        int hop = RES[r][1], window = RES[r][2], bins = window / 2 + 1;
        int frames = (n + window + hop) / hop + 1;
        float *s = malloc(sizeof(float) * frames * bins), *m = malloc(sizeof(float) * frames * bins);
        int count = stft_magnitude(ys, n, hop, window, tables->cos[r], tables->sin[r], s);
        stft_magnitude(target, n, hop, window, tables->cos[r], tables->sin[r], m);
        float diff = 0.0f, norm = 0.0f, logs = 0.0f;
        for (int i = 0; i < count * bins; i++) {
            diff += (m[i] - s[i]) * (m[i] - s[i]);
            norm += m[i] * m[i];
            logs += fabsf(logf(s[i]) - logf(m[i]));
        }
        total += sqrtf(diff) / sqrtf(norm) + logs / (count * bins);
        free(s);
        free(m);
    }
    free(ys);
    return total / 3.0f;
}

// MAIN ============================================================================================

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: models DIR\n");
        return 1;
    }
    const char *dir = argv[1];
    double t;

    // one-pole forward (no AD: the compiled filter)
    {
        Array c = load(dir, "one_pole_forward", "in0"), xs = load(dir, "one_pole_forward", "in1"), s0 = load(dir, "one_pole_forward", "in2");
        Array want = load(dir, "one_pole_forward", "out0");
        int n = xs.len;
        float *ys = malloc(sizeof(float) * n);
        MEDIAN_TIME(t, one_pole(c.data[0], xs.data, s0.data[0], ys, n));
        row("one-pole low-pass, 48k samples", t, error(ys, want));
    }

    // one-pole MSE gradient
    {
        Array c = load(dir, "one_pole_grad", "in0"), xs = load(dir, "one_pole_grad", "in1");
        Array tg = load(dir, "one_pole_grad", "in2"), s0 = load(dir, "one_pole_grad", "in3");
        Array w_loss = load(dir, "one_pole_grad", "out0"), w_dc = load(dir, "one_pole_grad", "out1");
        Array w_ds = load(dir, "one_pole_grad", "out2"), w_dx = load(dir, "one_pole_grad", "out3");
        int n = xs.len;
        float dc, ds, *dx = malloc(sizeof(float) * n), loss = 0.0f;
        MEDIAN_TIME(t, {
            dc = 0.0f;
            ds = 0.0f;
            memset(dx, 0, sizeof(float) * n);
            __enzyme_autodiff((void *)one_pole_loss, enzyme_dup, c.data, &dc, enzyme_dup, xs.data, dx, enzyme_const, tg.data, enzyme_dup, s0.data, &ds, enzyme_const, n);
            loss = one_pole_loss(c.data, xs.data, tg.data, s0.data, n);
        });
        double err = fmax(fmax(error(&loss, w_loss), error(&dc, w_dc)), fmax(error(&ds, w_ds), error(dx, w_dx)));
        // time the gradient alone (the loss call above is for checking)
        MEDIAN_TIME(t, {
            dc = 0.0f;
            ds = 0.0f;
            memset(dx, 0, sizeof(float) * n);
            __enzyme_autodiff((void *)one_pole_loss, enzyme_dup, c.data, &dc, enzyme_dup, xs.data, dx, enzyme_const, tg.data, enzyme_dup, s0.data, &ds, enzyme_const, n);
        });
        row("one-pole MSE gradient, 48k samples", t, err);
    }

    // EQ + drive, multi-resolution STFT loss gradient
    {
        Array p[7];
        for (int k = 0; k < 7; k++) {
            char which[8];
            snprintf(which, sizeof which, "in%d", k);
            p[k] = load(dir, "eq_drive_stft_grad", which);
        }
        Array want[7];
        for (int k = 0; k < 7; k++) {
            char which[8];
            snprintf(which, sizeof which, "out%d", k);
            want[k] = load(dir, "eq_drive_stft_grad", which);
        }
        // inputs: logf, g10, drive, xs, target, s0a, s0b; outputs: loss, d logf, d g10, d drive, d s0a, d s0b, d xs
        int n = p[3].len;
        Tables tables;
        for (int r = 0; r < 3; r++) twiddles(RES[r][2], (float **)&tables.cos[r], (float **)&tables.sin[r]);
        float g[5], *dx = malloc(sizeof(float) * n), loss = 0.0f;
        MEDIAN_TIME(t, {
            memset(g, 0, sizeof g);
            memset(dx, 0, sizeof(float) * n);
            __enzyme_autodiff((void *)eq_drive_loss, enzyme_dup, p[0].data, &g[0], enzyme_dup, p[1].data, &g[1], enzyme_dup, p[2].data, &g[2],
                              enzyme_dup, p[5].data, &g[3], enzyme_dup, p[6].data, &g[4], enzyme_dup, p[3].data, dx, enzyme_const, p[4].data,
                              enzyme_const, &tables, enzyme_const, n);
        });
        loss = eq_drive_loss(p[0].data, p[1].data, p[2].data, p[5].data, p[6].data, p[3].data, p[4].data, &tables, n);
        double err = fmax(error(&loss, want[0]), error(dx, want[6]));
        for (int k = 0; k < 5; k++) err = fmax(err, error(&g[k], want[k + 1]));
        row("EQ + drive, multi-resolution STFT loss gradient, 2048 samples", t, err);
    }

    // rfft -> gain -> irfft -> dense -> tanh gradient
    {
        Array x = load(dir, "spectral_model_grad", "in0"), gain = load(dir, "spectral_model_grad", "in1"), w = load(dir, "spectral_model_grad", "in2");
        Array w_loss = load(dir, "spectral_model_grad", "out0"), w_dg = load(dir, "spectral_model_grad", "out1"), w_dw = load(dir, "spectral_model_grad", "out2");
        int bins = gain.len, len = (bins - 1) * 2, width = w.len / len, batch = x.len / len;
        float *tc, *ts;
        twiddles(len, &tc, &ts);
        float *dg = malloc(sizeof(float) * bins), *dw = malloc(sizeof(float) * w.len);
        MEDIAN_TIME(t, {
            memset(dg, 0, sizeof(float) * bins);
            memset(dw, 0, sizeof(float) * w.len);
            __enzyme_autodiff((void *)spectral_loss, enzyme_const, x.data, enzyme_dup, gain.data, dg, enzyme_dup, w.data, dw, enzyme_const, tc,
                              enzyme_const, ts, enzyme_const, batch, enzyme_const, len, enzyme_const, width);
        });
        float loss = spectral_loss(x.data, gain.data, w.data, tc, ts, batch, len, width);
        double err = fmax(error(&loss, w_loss), fmax(error(dg, w_dg), error(dw, w_dw)));
        row("rfft -> gain -> irfft -> dense -> tanh gradient, 256 x 1024", t, err);
    }
    return 0;
}
