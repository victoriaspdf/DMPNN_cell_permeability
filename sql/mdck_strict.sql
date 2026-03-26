CREATE TABLE mdck_strict AS
SELECT
    a.activity_id,
    a.assay_id,
    a.molregno,
    cs.canonical_smiles AS smiles,
    a.standard_type,
    a.standard_value AS mdck_value,
    a.standard_units AS units,
    ass.description AS assay_description
FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN compound_structures cs ON a.molregno = cs.molregno
WHERE ass.assay_type = 'A'
  AND LOWER(ass.description) LIKE '%mdck%'
  AND a.standard_value IS NOT NULL
  AND a.standard_units IS NOT NULL
  AND LOWER(a.standard_units) LIKE '%m/s%'
  AND LOWER(ass.description) NOT LIKE '%mdr1%'
  AND LOWER(ass.description) NOT LIKE '%p-gp%'
  AND LOWER(ass.description) NOT LIKE '%pgp%'
  AND LOWER(ass.description) NOT LIKE '%p-glycoprotein%'
  AND LOWER(ass.description) NOT LIKE '%mrp1%'
  AND LOWER(ass.description) NOT LIKE '%mrp2%'
  AND LOWER(ass.description) NOT LIKE '%bcrp%'
  AND LOWER(ass.description) NOT LIKE '%basolateral to apical%'
  AND a.assay_id IN (
      SELECT assay_id
      FROM activities
      WHERE standard_value IS NOT NULL
      GROUP BY assay_id
      HAVING COUNT(DISTINCT standard_value) > 2
  );




