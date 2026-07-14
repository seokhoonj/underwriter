# underwriter

Automated underwriting from insurance claim history. A [polars](https://pola.rs)
pipeline that cleanses Korean credit-information (ICIS) claim history, maps
diagnosis codes to representative diseases, and aggregates per-insured
underwriting inputs — then, given a rule set, matches those inputs and combines
the per-disease decisions into a final per-insured decision.

The engine is data-agnostic: the rule set is supplied at run time, so it is not
tied to any one insurer.

> **Sibling R package:** [seokhoonj/underwriter-r](https://github.com/seokhoonj/underwriter-r).
> The two share one contract; the Python front half is validated to reproduce the
> R output exactly.

## Install

```bash
pip install "git+https://github.com/seokhoonj/underwriter.git"
pip install "underwriter[pandas] @ git+https://github.com/seokhoonj/underwriter.git"  # accept/return pandas too
```

Python 3.10+. polars only; pandas is optional.

## Quick start

The front half turns raw claim lines into per-insured underwriting inputs. The
example uses the bundled synthetic generators, so it runs with no data files.

```python
import underwriter as uw

# synthetic ICIS claims + a small KCD -> disease lookup (stand-ins for real data)
claims = uw.make_icis(n_insured=300, seed=0)
disease = uw.make_disease_table()

cleaned = uw.filter_latest_inquiry(uw.clean_icis(claims))   # parse, reconcile stays, dedup
mapped = uw.map_disease(uw.melt_kcd(cleaned), disease)       # -> kcd_main + scope flags
aggregated = uw.aggregate_disease(mapped)                    # one row per (id, disease)

aggregated.head()
# every insured survives every stage: no id is ever dropped
assert aggregated["id"].n_unique() == claims["id"].n_unique()
```

See [examples/notebooks/01.ICIS-Claim-Data-Processing.ipynb](examples/notebooks/01.ICIS-Claim-Data-Processing.ipynb).

## Pipeline

| stage | function | in → out |
|---|---|---|
| cleanse | `clean_icis`, `filter_latest_inquiry` | raw claim lines → one clean row per line, latest inquiry |
| melt | `melt_kcd` | wide `kcd0..kcd4` → one row per diagnosis code |
| map | `map_disease` | code → representative disease + lookback / scope flags |
| aggregate | `aggregate_disease` | → one row per `(id, disease)`: counts and elapsed days |

**No insured left behind.** A line with no code becomes `VACANT`, an unreadable
code `IRREGULAR`, a valid-but-unmapped code `UNMAPPED`, and an insured whose
whole history has aged out is carried on an `EXPIRED` placeholder — so every
insured reaches a decision.

## Rule engine

With a rule set (disease rules, sentinel catch-alls, and decision tables),
`Underwriter(rulebook).underwrite(aggregated)` band-matches each input and
combines the per-disease decisions into one decision per insured (accept /
conditional / decline / refer). The rule set is a company resource supplied at
run time and is not shipped with the package.

```python
book = uw.Rulebook.from_excel("rulebook.xlsx")   # your own rule workbook
decision = uw.Underwriter(book).underwrite(aggregated)
decision.tabulate()                               # decision distribution per coverage
```
