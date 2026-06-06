-- Healthcare Readmission Analysis
-- SQL queries for exploratory analysis
-- To be expanded in future iterations
-- dialect: postgresql

-- Total encounters by readmission class
SELECT readmitted, COUNT(*) AS total_encounters
FROM diabetes_data
GROUP BY readmitted
ORDER BY total_encounters DESC;

-- Average time in hospital by readmission class
SELECT readmitted, ROUND(AVG(time_in_hospital), 2) AS avg_days
FROM diabetes_data
GROUP BY readmitted
ORDER BY avg_days DESC;

-- Top 10 medical specialties by encounter count
SELECT medical_specialty, COUNT(*) AS total_encounters
FROM diabetes_data
GROUP BY medical_specialty
ORDER BY total_encounters DESC
LIMIT 10;

-- Average medications by age group
SELECT age, ROUND(AVG(num_medications), 2) AS avg_medications
FROM diabetes_data
GROUP BY age
ORDER BY avg_medications DESC;