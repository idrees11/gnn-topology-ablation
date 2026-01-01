
# GNN Challenge Leaderboard

This leaderboard shows the top submissions for the GNN challenge.  
Participants submit predictions (`sample_submission.csv`) and the F1-score is computed against the hidden test labels.

| Rank | Participant | Submission File | F1 Score | Date |
|------|------------|----------------|----------|------|
| 1    | Alice      | alice_v1.csv   | 0.8123   | 2026-01-01 |
| 2    | Bob        | bob_model.csv  | 0.7985   | 2026-01-01 |
| 3    | Charlie    | charlie.csv    | 0.7738   | 2026-01-01 |

---

## **How it works**

1. Participants download the public repo, train their model, and generate a submission CSV.  
2. Organizers use `scoring_script.py` to calculate F1-score against `private_data/test_labels.csv`.  
3. The leaderboard is updated manually (or automatically via GitHub Actions).  

> **Note**: Only organizers have access to the hidden test labels. Participants cannot see them.

