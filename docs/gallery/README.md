# Visual Gallery

The [gallery manifest](manifest.v1.json) defines two generated examples from
each of Trace's 11 visual domains. Regenerate the gallery with:

```bash
python scripts/generate_release_gallery.py
```

The [README montage manifest](paper-domain-montage.v1.json) defines the
committed examples used in the 11-domain montage. Regenerate it with:

```bash
python scripts/generate_paper_domain_montage.py
```
