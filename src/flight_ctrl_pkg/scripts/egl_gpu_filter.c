#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifndef EGL_CUDA_DEVICE_NV
#define EGL_CUDA_DEVICE_NV 0x323A
#endif

typedef __eglMustCastToProperFunctionPointerType (*pfn_eglGetProcAddress_t)(const char *);
typedef EGLBoolean (*pfn_eglQueryDevicesEXT_t)(EGLint, EGLDeviceEXT *, EGLint *);
typedef EGLBoolean (*pfn_eglQueryDeviceAttribEXT_t)(EGLDeviceEXT, EGLint, EGLAttrib *);

static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static void *g_egl_handle = NULL;
static pfn_eglGetProcAddress_t g_real_eglGetProcAddress = NULL;
static pfn_eglQueryDevicesEXT_t g_real_eglQueryDevicesEXT = NULL;
static pfn_eglQueryDeviceAttribEXT_t g_real_eglQueryDeviceAttribEXT = NULL;

static int is_ascii_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

__attribute__((constructor))
static void sanitize_cuda_visible_devices_ctor(void) {
    /* Belt-and-suspenders: if caller forgot gpu3_guard.sh and left CUDA_VISIBLE_DEVICES
       unset or containing token "3" / GPU-3 UUID (with optional ASCII whitespace),
       strip it (defaulting to "2" when non-whitespace input leaves no valid tokens). */
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (cvd == NULL) {
        setenv("CUDA_VISIBLE_DEVICES", "2", 1);
        return;
    }
    if (cvd[0] == '\0') {
        return; /* Explicit empty string -> 0 CUDA devices */
    }
    size_t len = strlen(cvd);
    int has_non_ws = 0;
    for (size_t i = 0; i < len; ++i) {
        if (!is_ascii_ws(cvd[i])) {
            has_non_ws = 1;
            break;
        }
    }
    if (!has_non_ws) {
        return; /* Whitespace-only -> preserve explicit empty (0 CUDA devices) semantics */
    }

    char buf[512];
    char out[512];
    if (len >= sizeof(buf)) {
        setenv("CUDA_VISIBLE_DEVICES", "2", 1);
        return;
    }
    memcpy(buf, cvd, len + 1);
    size_t out_len = 0;
    out[0] = '\0';
    int modified = 0;

    char *cursor = buf;
    for (;;) {
        char *comma = strchr(cursor, ',');
        if (comma != NULL) {
            *comma = '\0';
        }

        char *tok = cursor;
        while (*tok != '\0' && is_ascii_ws(*tok)) {
            ++tok;
            modified = 1;
        }
        size_t tlen = strlen(tok);
        while (tlen > 0 && is_ascii_ws(tok[tlen - 1])) {
            tok[--tlen] = '\0';
            modified = 1;
        }

        if (tlen == 0) {
            modified = 1;
        } else if (strcmp(tok, "3") == 0 ||
                   strcmp(tok, "GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5") == 0) {
            modified = 1;
        } else {
            if (out_len > 0) {
                if (out_len + 1 >= sizeof(out)) {
                    setenv("CUDA_VISIBLE_DEVICES", "2", 1);
                    return;
                }
                out[out_len++] = ',';
            }
            if (out_len + tlen >= sizeof(out)) {
                setenv("CUDA_VISIBLE_DEVICES", "2", 1);
                return;
            }
            memcpy(out + out_len, tok, tlen);
            out_len += tlen;
            out[out_len] = '\0';
        }

        if (comma == NULL) {
            break;
        }
        cursor = comma + 1;
    }

    if (modified || out_len == 0) {
        setenv("CUDA_VISIBLE_DEVICES", (out_len > 0) ? out : "2", 1);
    }
}

static void init_real_egl(void) {
    /* Prefer direct handle to libEGL.so.1 so multiple LD_PRELOAD entries never chain to each other. */
    g_egl_handle = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (g_egl_handle == NULL) {
        g_egl_handle = dlopen("libEGL.so.1", RTLD_LAZY | RTLD_LOCAL);
    }
    if (g_egl_handle != NULL) {
        g_real_eglGetProcAddress = (pfn_eglGetProcAddress_t)dlsym(g_egl_handle, "eglGetProcAddress");
    }
    if (g_real_eglGetProcAddress == NULL || g_real_eglGetProcAddress == eglGetProcAddress) {
        g_real_eglGetProcAddress = (pfn_eglGetProcAddress_t)dlsym(RTLD_NEXT, "eglGetProcAddress");
    }
    if (g_real_eglGetProcAddress != NULL && g_real_eglGetProcAddress != eglGetProcAddress) {
        pfn_eglQueryDevicesEXT_t qdev =
            (pfn_eglQueryDevicesEXT_t)g_real_eglGetProcAddress("eglQueryDevicesEXT");
        if (qdev != eglQueryDevicesEXT) {
            g_real_eglQueryDevicesEXT = qdev;
        }
        g_real_eglQueryDeviceAttribEXT =
            (pfn_eglQueryDeviceAttribEXT_t)g_real_eglGetProcAddress("eglQueryDeviceAttribEXT");
    }
}

EGLAPI EGLBoolean EGLAPIENTRY eglQueryDevicesEXT(
    EGLint max_devices,
    EGLDeviceEXT *devices,
    EGLint *num_devices)
{
    pthread_once(&g_init_once, init_real_egl);

    if (num_devices == NULL) {
        return EGL_FALSE;
    }
    /* Per EGL_EXT_device_enumeration: max_devices <= 0 is invalid only when devices != NULL */
    if (devices != NULL && max_devices <= 0) {
        return EGL_FALSE;
    }
    if (g_real_eglQueryDevicesEXT == NULL) {
        *num_devices = 0;
        return EGL_FALSE;
    }

    EGLint raw_total = 0;
    if (g_real_eglQueryDevicesEXT(0, NULL, &raw_total) != EGL_TRUE) {
        return EGL_FALSE;
    }
    if (raw_total <= 0) {
        *num_devices = 0;
        return EGL_TRUE;
    }

    EGLDeviceEXT stack_raw[64];
    EGLDeviceEXT stack_filt[64];
    EGLDeviceEXT *raw_devs = stack_raw;
    EGLDeviceEXT *filt_devs = stack_filt;
    if (raw_total > 64) {
        raw_devs = (EGLDeviceEXT *)calloc((size_t)raw_total, sizeof(EGLDeviceEXT));
        filt_devs = (EGLDeviceEXT *)calloc((size_t)raw_total, sizeof(EGLDeviceEXT));
        if (raw_devs == NULL || filt_devs == NULL) {
            free(raw_devs);
            free(filt_devs);
            return EGL_FALSE;
        }
    }

    EGLint raw_count = raw_total;
    EGLBoolean ok = g_real_eglQueryDevicesEXT(raw_total, raw_devs, &raw_count);
    if (ok != EGL_TRUE) {
        if (raw_total > 64) {
            free(raw_devs);
            free(filt_devs);
        }
        return EGL_FALSE;
    }

    EGLint filt_count = 0;
    if (g_real_eglQueryDeviceAttribEXT != NULL) {
        for (EGLint i = 0; i < raw_count; ++i) {
            EGLAttrib cuda_id = -1;
            if (g_real_eglQueryDeviceAttribEXT(raw_devs[i], EGL_CUDA_DEVICE_NV, &cuda_id) == EGL_TRUE) {
                filt_devs[filt_count++] = raw_devs[i];
            }
        }
    }

    if (devices == NULL) {
        /* Count query: ignore max_devices, report total filtered count */
        *num_devices = filt_count;
    } else {
        EGLint to_copy = (filt_count < max_devices) ? filt_count : max_devices;
        for (EGLint i = 0; i < to_copy; ++i) {
            devices[i] = filt_devs[i];
        }
        *num_devices = to_copy;
    }

    if (raw_total > 64) {
        free(raw_devs);
        free(filt_devs);
    }
    return EGL_TRUE;
}

EGLAPI __eglMustCastToProperFunctionPointerType EGLAPIENTRY eglGetProcAddress(const char *procname) {
    pthread_once(&g_init_once, init_real_egl);
    if (procname != NULL && strcmp(procname, "eglQueryDevicesEXT") == 0) {
        return (__eglMustCastToProperFunctionPointerType)eglQueryDevicesEXT;
    }
    if (g_real_eglGetProcAddress != NULL) {
        return g_real_eglGetProcAddress(procname);
    }
    return NULL;
}
