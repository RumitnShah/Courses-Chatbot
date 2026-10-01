# Embedding Guide

Read this before adding or re-creating embeddings with `app_croq.py`.
The chatbot (`QA_croq.py`) depends on the settings below, so a mismatch will
break search or show wrong source links.

## 1. Settings that must NOT change (unless you rebuild everything)

| Setting | Current value | Why it matters |
|---|---|---|
| Embedding model | `intfloat/e5-large-v2` | `QA_croq.py` embeds questions with the same model. Different models produce incompatible vectors. |
| Vector dimension | `1024` | Fixed when the Pinecone index is created. |
| Similarity metric | `cosine` | Fixed when the Pinecone index is created. |
| Index name | `course-database` | Hard-coded in both files. |
| `normalize_embeddings` | `True` | Keep it identical in both files. |
| Text metadata key | `text` | `QA_croq.py` reads the chunk text from `metadata["text"]`. |

If you ever switch the embedding model, create a **new index** with the new
model's dimension, re-embed **all** PDFs into it, and update `EMBEDDING_MODEL`
and `INDEX_NAME` in `QA_croq.py`.

## 2. Metadata stored today

Each chunk currently has only:

```python
{
    "source": "B.Tech Mechanical Engineering Course Structure",  # display name
    "source_url": "https://drive.google.com/file/d/.../view",     # link shown to users
    "text": "...chunk content...",
}
```

Current index contents: 676 chunks from 3 PDFs (Computer Science & Engineering,
Mechanical Engineering, Electronics & Communication Engineering).

## 3. Metadata to add for future embeddings

Add these fields in the `doc.metadata = {...}` block of `app_croq.py`:

```python
doc.metadata = {
    # --- required (used by QA_croq.py today) ---
    "text": doc.page_content,
    "source": "B.Tech Mechanical Engineering Course Structure",
    "source_url": web_url,

    # --- recommended for future use ---
    "program": "Mechanical",          # Computer / Mechanical / Electronics / ...
    "program_code": "ME",              # short code: CSE, ME, ECE, ...
    "file_name": "Mechanical.pdf",
    "page": page_number,               # 1-based page the chunk came from
    "chunk_index": i,                  # order of the chunk inside the PDF
    "semester": 4,                     # if the chunk belongs to one semester (else omit)
    "content_type": "course_structure",# or "course_syllabus", "electives", ...
    "course_codes": ["16MI205T", "16MI207T"],  # codes found in the chunk
    "academic_year": "2020-21",        # syllabus version ("w.e.f." year in the PDF)
    "embedding_model": "intfloat/e5-large-v2",
    "ingested_at": "2026-10-01",
}
```

What each recommended field enables:

- **`program` / `program_code`**: filter searches to the program the user asked
  about, and pick the correct source link without guessing from names.
- **`page`**: link users to the exact page of the PDF (use a host that supports
  `#page=N`, e.g. the PDF stored in this GitHub repo; Google Drive's viewer does not).
- **`chunk_index`**: lets the app join a semester table that is split across two
  chunks reliably (today this is inferred from the text overlap).
- **`semester`**: answer "semester N" questions by filtering instead of searching.
- **`course_codes`**: exact lookup of a course by its code.
- **`academic_year`**: keep old and new syllabus versions apart.
- **`embedding_model` / `ingested_at`**: know how and when each chunk was created.

Pinecone metadata values must be strings, numbers, booleans or lists of strings,
and the whole metadata for one chunk must stay under 40 KB.

## 4. Changes needed in `app_croq.py` for this

1. **Keep page numbers.** Today all pages are joined into one string, so page
   information is lost. Create one `Document` per page instead:
   ```python
   docs = [
       Document(page_content=page.extract_text() or "", metadata={"page": i + 1})
       for i, page in enumerate(pdf_reader.pages)
   ]
   ```
   Then split `docs` and copy `page` into each chunk's metadata.
2. **Use stable IDs** so re-running a PDF replaces its chunks instead of
   duplicating them, e.g. `f"{program_code}-p{page}-c{chunk_index}"`, and pass
   them with `ids=` to `PineconeVectorStore.from_documents(...)`.
3. **Delete old chunks before re-embedding a PDF.** With the ID prefix above:
   ```python
   old_ids = [i for page in index.list(prefix="ME-") for i in page]
   if old_ids:
       index.delete(ids=old_ids)
   ```
   The chunks stored today have random IDs, so to replace them you must delete
   them first (or re-create the whole index).
4. **Check the extracted text.** The Mechanical PDF currently extracts with
   symbols like `!` and `'` between words (e.g. `Course!Name:!Fluid!Mechanics`).
   Print a few chunks before uploading; if they look like this, clean them
   (e.g. `re.sub(r"[!'\"$]+", " ", text)`) or try a different PDF reader
   such as `pdfplumber`.
5. **Set `source` and `web_url` per PDF.** They are hard-coded for the
   Mechanical PDF; update both for every PDF you embed.

## 5. Things QA_croq.py expects

- The `source` name must contain **Computer**, **Mechanical** or
  **Electronics**: the source link is chosen by matching these words with
  the program in the question (`PROGRAM_PATTERNS` in `QA_croq.py`).
- **Adding a new program** (e.g. Civil): use a `source` name containing
  "Civil" and add an entry to `PROGRAM_PATTERNS`, for example
  `"Civil": r"\bcivil\b"`.
- The keyword search loads all chunks from Pinecone and caches them for 6 hours.
  After uploading new embeddings, **reboot the app** on Streamlit Cloud (or
  restart it locally) to use them straight away.
- e5 models work best with `"passage: "` before stored text and `"query: "`
  before questions. If you start adding the `passage: ` prefix, re-embed **all**
  PDFs and also add the `query: ` prefix in `QA_croq.py`; never mix the two styles.

## 6. Checklist for adding a new PDF

1. Put the PDF in `Revised Syllabus/` and upload it to Drive (or GitHub) for the link.
2. In `app_croq.py`, set `source`, `web_url` and the new metadata fields.
3. Run `python app_croq.py "Revised Syllabus/<File>.pdf"`.
4. Check the vector count in the Pinecone console went up as expected.
5. Reboot the Streamlit app and ask a question about the new PDF.
6. Check the answer and the source link.
