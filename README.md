# NFL Team Success Predictor & Offseason Simulator

![NFL Analytics](https://img.shields.io/badge/Sports-Analytics-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Web App](https://img.shields.io/badge/Web-App-green)

A comprehensive system for predicting NFL team success and simulating offseason roster management through machine learning and web technologies.

## Features

### Predictive Modeling
- Developed TensorFlow models including RNNs to forecast team success
- Analyzed position investment and player performance data
- Achieved **84% R² accuracy** for both team and player success predictions

### Offseason Simulator Web App
- **Frontend**: Interactive React interface
- **Backend**: Flask API service
- **Database**: MongoDB for flexible data storage
- Features:
  - Player signing simulations
  - Rookie draft scenarios
  - Trade management tools

### Data Pipeline
- Scraped 15 years of NFL historical data using Beautiful Soup
- Cleaned and processed data with Pandas
- Stored in MongoDB with integrity checks
- Dataset includes:
  - Team investment patterns
  - Player performance metrics
  - Salary cap information

## Technology Stack
- **Machine Learning**: TensorFlow, Keras, Scikit-learn
- **Web Framework**: Flask (Python backend)
- **Frontend**: React.js
- **Database**: MongoDB
- **Data Processing**: Pandas, NumPy
- **Web Scraping**: Beautiful Soup, Requests
