# BioNewK: BioPortal Integration for MedRAGChecker

This directory contains the BioPortal knowledge graph integration for the MedRAGChecker project, enabling ontology-based claim verification using 1000+ biomedical ontologies.

## Overview

This implementation extends the MedRAGChecker framework with:
1. **BioPortal KG Scorer**: Entity linking and ontology-based verification
2. **Hybrid KG Scorer**: Combines DRKG and BioPortal for best coverage
3. **Student RAGChecker**: Complete end-to-end evaluation pipeline with KG integration

## Files

- `eval_student_ragchecker_kg.py` - Main evaluation script with full RAGChecker metrics
- `bioportal_kg_scorer.py` - BioPortal-only KG scorer
- `hybrid_kg_scorer.py` - Hybrid DRKG + BioPortal scorer
- `quick_start_bioportal.py` - Quick start demo
- `BIOPORTAL_MIGRATION_GUIDE.md` - Detailed migration guide
- `results_text-8.json` - Sample data for testing

## Quick Start

### 1. Get BioPortal API Key

1. Sign up at https://bioportal.bioontology.org/
2. Get your API key from account settings
3. Set environment variable:
```bash
export BIOPORTAL_API_KEY="your-api-key-here"
```

### 2. Test BioPortal Connection

```bash
python quick_start_bioportal.py
```

### 3. Run Student RAGChecker Evaluation

#### Without KG (baseline - NLI only)

```bash
python eval_student_ragchecker_kg.py \
    --results_path ./results_text-8.json \
    --extractor_dir ../runs/extractor_sft \
    --checker_dir ../runs/checker_sft \
    --base_model_extractor /path/to/base/model \
    --base_model_checker /path/to/base/model \
    --kg_mode none \
    --out_json ./output/baseline_results.json \
    --out_csv ./output/baseline_metrics.csv
```

#### With BioPortal KG

```bash
python eval_student_ragchecker_kg.py \
    --results_path ./results_text-8.json \
    --extractor_dir ../runs/extractor_sft \
    --checker_dir ../runs/checker_sft \
    --base_model_extractor /path/to/base/model \
    --base_model_checker /path/to/base/model \
    --kg_mode bioportal \
    --bioportal_key $BIOPORTAL_API_KEY \
    --bioportal_ontologies SNOMEDCT,MESH,RXNORM,DOID \
    --out_json ./output/bioportal_results.json \
    --out_csv ./output/bioportal_metrics.csv
```

#### With Hybrid Mode (DRKG + BioPortal)

```bash
python eval_student_ragchecker_kg.py \
    --results_path ./results_text-8.json \
    --extractor_dir ../runs/extractor_sft \
    --checker_dir ../runs/checker_sft \
    --base_model_extractor /path/to/base/model \
    --base_model_checker /path/to/base/model \
    --kg_mode hybrid \
    --bioportal_key $BIOPORTAL_API_KEY \
    --out_json ./output/hybrid_results.json \
    --out_csv ./output/hybrid_metrics.csv
```

## Data Format

### Input Format (results_text-8.json)

```json
{
  "results": [
    {
      "query_id": "25429730",
      "query": "Are ILC2s increased in chronic rhinosinusitis?",
      "response": "Yes, ILC2s are increased in chronic rhinosinusitis.",
      "response_claims": [
        ["ILC2s", "are increased in", "chronic rhinosinusitis"]
      ],
      "retrieved_context": [
        {"text": "ILC2s are elevated in patients with CRSwNP..."}
      ]
    }
  ]
}
```

### Output Format

```json
{
  "config": {
    "kg_mode": "bioportal",
    "threshold": 0.6
  },
  "metrics": {
    "overall_f1": 0.85,
    "overall_precision": 0.88,
    "overall_recall": 0.82,
    "faithfulness": 0.89,
    "hallucination": 0.05,
    "kg_consistency": 0.76,
    "kg_coverage": 0.68
  },
  "results": [
    {
      "query_id": "25429730",
      "claims": ["ILC2s are increased in chronic rhinosinusitis."],
      "verification_results": [
        {
          "claim_text": "ILC2s are increased in chronic rhinosinusitis.",
          "nli_label": "entailed",
          "p_entailed": 0.92,
          "kg_score": 0.78,
          "kg_status": "ok"
        }
      ]
    }
  ]
}
```

## Metrics Explained

### Overall Metrics
- **F1**: Harmonic mean of precision and recall
- **Precision**: Faithfulness (supported claims / total claims)
- **Recall**: Claim recall (covered gold claims / total gold claims)

### Retriever Metrics
- **Claim Recall**: Percentage of gold claims covered by extracted claims
- **Context Precision**: Percentage of retrieved passages actually used

### Generator Metrics
- **Faithfulness**: Percentage of claims supported by evidence
- **Hallucination**: Percentage of claims contradicted by evidence
- **Context Utilization**: Percentage of claims using retrieved context

### KG Metrics
- **KG Consistency**: Average KG score for claims with KG support
- **KG Coverage**: Percentage of claims that can be verified via KG

## Advanced Usage

### Batch Processing on Remote Server

For remote execution (e.g., on PSC):

```bash
# On remote server
cd /ocean/projects/med230010p/yji3/MedicalRagChecker/BioNewK

# Submit SLURM job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ragchecker_kg
#SBATCH --output=logs/ragchecker_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

module load cuda/11.8
source activate medrag

python eval_student_ragchecker_kg.py \
    --results_path ./results_text-8.json \
    --extractor_dir ./runs/extractor_sft \
    --checker_dir ./runs/checker_sft \
    --base_model_extractor /ocean/projects/med230010p/shared/models/Meditron3-8B \
    --base_model_checker /ocean/projects/med230010p/shared/models/Meditron3-8B \
    --kg_mode bioportal \
    --bioportal_key $BIOPORTAL_API_KEY \
    --out_json ./output/results_\${SLURM_JOB_ID}.json \
    --out_csv ./output/metrics_\${SLURM_JOB_ID}.csv
EOF
```

### Customizing Ontologies

Different medical domains benefit from different ontology selections:

```bash
# For drug-disease claims
--bioportal_ontologies RXNORM,DOID,SNOMEDCT,MESH

# For gene-disease claims
--bioportal_ontologies HGNC,GO,DOID,OMIM

# For side effects
--bioportal_ontologies MEDDRA,SNOMEDCT,MESH

# For cancer research
--bioportal_ontologies NCIT,DOID,GO
```

## Integration with Existing Pipeline

To integrate with your existing evaluation scripts:

```python
# Import the KG scorer
from BioNewK.bioportal_kg_scorer import BioPortalKGScorer

# Initialize
scorer = BioPortalKGScorer(
    api_key="YOUR_API_KEY",
    ontologies=["SNOMEDCT", "MESH", "RXNORM", "DOID"],
    cache_dir="./cache"
)

# Score a claim triple
result = scorer.score_claim(
    subject="metformin",
    relation="treats",
    object="diabetes"
)

print(f"KG Score: {result.final_score:.3f}")
print(f"Evidence: {result.evidence}")
```

## Troubleshooting

### Issue: API Rate Limiting
**Solution**: The scorer includes automatic rate limiting. For heavy usage, enable caching:
```bash
--bioportal_cache ./cache
```

### Issue: No Entities Found
**Possible causes**:
- Entity mention too specific
- Wrong ontology selection
- Uncommon terminology

**Solution**: Use more ontologies or increase candidate limit in code:
```python
linker.link_entity(mention, top_k=10)  # Default is 5
```

### Issue: Low KG Coverage
**Possible causes**:
- Claims not in triple format
- Entities not in selected ontologies

**Solution**:
1. Ensure claims are in `[subject, relation, object]` format
2. Use broader ontology selection
3. Try hybrid mode for better coverage

## Performance Tips

1. **Enable Caching**: Always use `--bioportal_cache` for repeated evaluations
2. **Batch Size**: Adjust `--batch_size` based on GPU memory
3. **Parallel Processing**: Run multiple evaluations in parallel with different data splits
4. **Ontology Selection**: Start with core ontologies (SNOMEDCT, MESH) and add domain-specific ones

## Citation

If you use this code, please cite:

```bibtex
@article{medragchecker2025,
  title={MedRAGChecker: Verifiable RAG for Biomedical Question Answering},
  author={...},
  journal={arXiv preprint arXiv:2601.06519},
  year={2025}
}
```

## Support

For issues or questions:
1. Check `BIOPORTAL_MIGRATION_GUIDE.md` for detailed documentation
2. Review BioPortal API docs: https://data.bioontology.org/documentation
3. Contact: yji3@psc.edu

## License

See parent directory LICENSE file.
