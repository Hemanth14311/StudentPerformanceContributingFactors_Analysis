# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("StudentPerformanceFactors.csv")
# to dispaly first five rows of data
data.head()
print("More information of data:\n",data.info())
print("Summary of numerical columns:\n",data.describe())
# Checking the null values in a data
data.isnull().sum()
# There is some missing values in columns Teacher_Quality, Parental_education_Level, and Distance_from_Home 
# by using simple imputer, I replace null values
from sklearn.impute import SimpleImputer
# If you want to impute categorical columns (e.g., with the most frequent value)
imputer_cat = SimpleImputer(strategy='most_frequent')
data[['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']] = imputer_cat.fit_transform(data[['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']])
# Check if there are any missing values left
print(data.isnull().sum())
study_exam = data[['Hours_Studied', 'Exam_Score', 'Gender']]

# Define conditions
conditions = [
    (study_exam['Exam_Score'] >= 90),
    (study_exam['Exam_Score'] >= 80) & (study_exam['Exam_Score'] < 90),
    (study_exam['Exam_Score'] >= 70) & (study_exam['Exam_Score'] < 80),
    (study_exam['Exam_Score'] >= 60) & (study_exam['Exam_Score'] < 70),
    (study_exam['Exam_Score'] >= 50) & (study_exam['Exam_Score'] < 60),
]

# Define corresponding grades
grades = ['O', 'A', 'B', 'C', 'Passed']

# Apply np.select with a default value
study_exam['grade'] = np.select(conditions, grades, default='Failed')

# Display first few rows
print(study_exam.head())
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))  
sns.barplot(data=study_exam, x='grade', y='Hours_Studied', hue='Gender', palette='muted')
plt.title("Hours Studied by Student Grades", fontsize=16, fontweight='bold')
plt.xlabel("Grade", fontsize=12)
plt.ylabel("Hours Studied", fontsize=12)
plt.show()
"""Possible Statistical Information: The mean and 
variance of hours studied for different grades and genders. 
Skewness can indicate if most students studied too 
little or too much."""

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.regplot(x='Hours_Studied', y='Exam_Score', data=data, scatter_kws={'s': 100}, line_kws={'color': 'red', 'lw': 2})
plt.title('Hours Studied vs Exam Score', fontsize=16, fontweight='bold')
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.show()
"""Possible Statistical Information: The mean and variance of both study hours and exam scores. Correlation can be inferred from the scatter plot, and skewness 
can help identify any outliers or asymmetries."""
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 6))  # Increase the figure size for better readability
# First subplot: Attendance vs Exam Score
plt.subplot(1, 2, 1)
sns.lineplot(data, x='Attendance', y='Exam_Score', color='b', marker='o', linewidth=2, markersize=6)
plt.title("Attendance Wise Marks", fontsize=16, fontweight='bold')
plt.xlabel('Attendance', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.grid(True)

# Second subplot: Attendance vs Exam Score by Gender
plt.subplot(1, 2, 2)
sns.lineplot(data,x='Attendance', y='Exam_Score', hue='Gender', palette='coolwarm', marker='o', linewidth=2, markersize=6)
plt.title("Gender-wise Attendance Marks", fontsize=16, fontweight='bold')
plt.xlabel('Attendance', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.legend(title='Gender', title_fontsize=13, loc='upper left', fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()
"""This shows the trend of how attendance affects the exam score. A negative or 
positive slope indicates the impact of attendance on performance."""
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(data, 
                x='Sleep_Hours', 
                y='Exam_Score', 
                size='Exam_Score', 
                hue='Exam_Score', 
                palette='viridis',  
                sizes=(50, 500),    
                marker='o',         
                edgecolor='black',  
                linewidth=0.7)      

plt.title("Exam Score vs Sleep Hours", fontsize=16, fontweight='bold')
plt.xlabel("Sleep Hours", fontsize=12)
plt.ylabel("Exam Score", fontsize=12)
plt.legend(title='Exam Score', title_fontsize=13, loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data, 
            x='Family_Income', 
            y='Exam_Score', 
            hue='Family_Income', 
            palette='coolwarm',  
            width=0.7,          
            fliersize=5,         
            linewidth=1.2)       
plt.title("Impact of Family Income on Student Marks", fontsize=14, fontweight='bold')
plt.xlabel("Family Income", fontsize=12)
plt.ylabel("Exam Score", fontsize=12)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Plot 2: Parental Education Level vs Exam Score
plt.subplot(1, 2, 2)
sns.boxplot(data, 
            x='Parental_Education_Level', 
            y='Exam_Score', 
            hue='Parental_Education_Level', 
            palette='coolwarm', 
            width=0.7,           
            fliersize=5,         
            linewidth=1.2)       
plt.title("Impact of Parental Education Level on Student Marks", fontsize=14, fontweight='bold')
plt.xlabel("Parental Education Level", fontsize=12)
plt.ylabel("Exam Score", fontsize=12)
plt.xticks(rotation=45)  
plt.tight_layout()
plt.show()
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))  
sns.barplot(data, 
            x='School_Type', 
            y='Exam_Score', 
            hue='School_Type', 
            palette='muted',     
            ci=None)            
plt.title("Performance Comparison: Private vs Public School Students", fontsize=16, fontweight='bold')
plt.xlabel("School Type", fontsize=12)
plt.ylabel("Average Exam Score", fontsize=12)
plt.legend(title="School Type", fontsize=12, title_fontsize='13', loc='upper right')
plt.xticks(rotation=0)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  
plt.show()
# Plotting histograms for numerical columns
numerical_cols = ['Hours_Studied', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Teacher_Quality', y='Exam_Score', palette='muted')
plt.title('Exam Score Distribution by Teacher Quality', fontsize=16, fontweight='bold')
plt.xlabel('Teacher Quality', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.show()
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Motivation_Level'], y=data['Exam_Score'])
plt.title('Motivation Level vs Exam Score')
plt.xticks(rotation=45)
plt.show()
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='School_Type', hue='Gender', palette='muted', dodge=False)
plt.title('School Type vs Gender Distribution', fontsize=16, fontweight='bold')
plt.xlabel('School Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()
sns.set_theme(style="whitegrid")
numerical_data = data[['Hours_Studied', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']]
sns.pairplot(numerical_data, hue="Exam_Score", palette='coolwarm', markers='o')
plt.title('Pairwise Relationships between Features', fontsize=16, fontweight='bold')
plt.show()
corr = data[['Hours_Studied', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
plt.show()
