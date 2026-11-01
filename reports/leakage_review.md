# Test leakage review

The official train/validation/test folders remain unchanged. `clean_test_list.txt` contains the test images retained for the clean evaluation.

- Official test images: 1200
- Visually reviewed near-duplicate test images excluded: 57 (nearest training-image pHash Hamming distance <= 4)
- Clean test images: 1143
- Visual review: all 57 reported test candidates at distances 2 or 4 were checked in contact sheets; each appears to show the same underlying photo as its nearest training match, with crop, resize, or color changes.
- Label conflict to review in source data: `data/test/Seborrheic Dermatitis/seborrheic_dermatitis_0748.jpg` appears to duplicate `data/train/Psoriasis/psoriasis_0557.jpg` despite the different class labels.

## Retained test images by class

| Class | Retained | Excluded |
|---|---:|---:|
| Alopecia Areata | 113 | 7 |
| Contact Dermatitis | 117 | 3 |
| Folliculitis | 107 | 13 |
| Head Lice | 117 | 3 |
| Lichen Planus | 114 | 6 |
| Male Pattern Baldness | 111 | 9 |
| Psoriasis | 116 | 4 |
| Seborrheic Dermatitis | 116 | 4 |
| Telogen Effluvium | 112 | 8 |
| Tinea Capitis | 120 | 0 |
