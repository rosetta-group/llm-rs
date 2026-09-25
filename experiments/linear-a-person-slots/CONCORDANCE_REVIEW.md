# Post-run concordance checks

The frozen literal concordance counts source word strings and source object keys.
Its `objects` field is not an independently deduplicated count of archaeological objects.

**Object key:** the corpus `parent_object`, falling back to document ID.
**Physical object:** a tablet identified across all faces and alternative catalogue names.

1. **A catalogue alias can inflate support.** `ma-di` has six occurrences under six object
   keys, but `PH (?)31a` and `PH 31a` both carry museum inventory **HM 1609**, the same
   GORILA reference and overlapping contents. The uncertain siglum record has no sign layer;
   the other record lacks a glyph text. Do not report these as two independent attestations.
   They remain separate source records in the literal output, with its quality flags.
2. **The preferred reading is an overlay.** `te-ja-re` has zero word-layer hits;
   `te-*56-re` has one, on HT117a. This is expected: the query never patches the corpus.
   The [source review](source-review.json) separately prefers the edition's AB57 reading.
3. **An additional role contrast needs collation.** `qi-tu-ne` occurs on HT7b as well as
   HT87 and HT117b. The pinned HT7b glyph layer places numeral 1 after it, unlike the two
   inspected headers. This is a useful next source-check target, not a new gold personal
   name or a third independently reviewed header. HT7b was not image-collated in this audit.
4. **Multiplicity stays visible.** `ku-*56-nu` has eight literal occurrences under seven
   object keys, including two entries on HT122a. `qa-A310-i` has four occurrences under
   three keys because HT8a/b are faces of one object. Neither token multiplicity nor a
   shared spelling establishes identity of people across documents.

These checks were made after the descriptive run and do not alter its frozen inputs or
outputs. No statistic, training label or kinship edge is derived from these counts. The
main audit's four faces belong to two identified objects, HT85 and HT117; the concordance
alias issue does not change its 43-row inventory. A future whole-corpus model needs a
reviewed catalogue-alias map before splitting or counting independent objects.
