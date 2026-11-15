/* kural-engine C interface. Link with -lkural_engine. */
#ifndef KURAL_H
#define KURAL_H
#include <stddef.h>
#include <stdint.h>

typedef struct kural_engine kural_engine;

/* Load a model file (model.kural). NULL on failure. */
kural_engine *kural_open(const char *path);

/* Parse a request. context: previous-request words separated by 0x1f, or NULL.
   Writes a JSON array of readings into out; returns its length (repeat with a bigger buffer if >= cap), -1 on error. */
int64_t kural_parse(const kural_engine *e, const char *text, const char *context, int32_t beam, int32_t keep,
                    char *out, size_t cap);

void kural_close(kural_engine *e);
#endif
