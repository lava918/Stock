import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def load_and_prepare_data(file_path):
    """Load and prepare Tesla stock data."""
    try:
        # Read the CSV file
        tesla = pd.read_csv('tesla.csv')
        
        # Convert date column to datetime
        tesla['Date'] = pd.to_datetime(tesla['Date'])
        
        # Print basic information about the dataset
        print(f'Dataframe contains stock prices between {tesla.Date.min()} and {tesla.Date.max()}')
        print(f'Total days = {(tesla.Date.max() - tesla.Date.min()).days} days')
        
        return tesla
    except FileNotFoundError:
        print(f"Error: Could not find file {'tesla.csv'}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_stock_visualization(tesla_df):
    """Create interactive stock price visualization."""
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Stock Price', 'Trading Volume'),
                       vertical_spacing=0.2,
                       row_heights=[0.7, 0.3])

    # Add stock price trace
    fig.add_trace(
        go.Scatter(x=tesla_df['Date'], 
                  y=tesla_df['Close'],
                  name='Close Price',
                  line=dict(color='#17BECF')),
        row=1, col=1
    )

    # Add volume trace
    fig.add_trace(
        go.Bar(x=tesla_df['Date'],
               y=tesla_df['Volume'],
               name='Volume',
               marker=dict(color='#7F7F7F')),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title='Tesla Stock Analysis',
        height=800,
        showlegend=True,
        xaxis2_title='Date',
        yaxis_title='Price (USD)',
        yaxis2_title='Volume'
    )

    return fig

def build_regression_model(tesla_df):
    """Build and evaluate linear regression model."""
    # Prepare features and target
    X = np.array(range(len(tesla_df))).reshape(-1, 1)  # Using index as feature
    y = tesla_df['Close'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred),
        'train_mse': mean_squared_error(y_train, train_pred),
        'test_mse': mean_squared_error(y_test, test_pred)
    }
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'metrics': metrics
    }

def plot_regression_results(results):
    """Plot actual vs predicted values."""
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter(
        x=results['X_train'].flatten(),
        y=results['y_train'],
        mode='markers',
        name='Training Data',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Add training predictions
    fig.add_trace(go.Scatter(
        x=results['X_train'].flatten(),
        y=results['train_pred'],
        mode='lines',
        name='Model Predictions',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Linear Regression: Actual vs Predicted Values',
        xaxis_title='Days',
        yaxis_title='Stock Price',
        showlegend=True
    )
    
    return fig

def main():
    # Load and prepare data
    tesla_df = load_and_prepare_data('tesla.csv')
    if tesla_df is None:
        return
    
    # Create stock visualization
    stock_fig = create_stock_visualization(tesla_df)
    stock_fig.show()
    
    # Build and evaluate model
    results = build_regression_model(tesla_df)
    
    # Print model metrics
    print("\nModel Evaluation Metrics:")
    print(f"Training R² Score: {results['metrics']['train_r2']:.4f}")
    print(f"Testing R² Score: {results['metrics']['test_r2']:.4f}")
    print(f"Training MSE: {results['metrics']['train_mse']:.4f}")
    print(f"Testing MSE: {results['metrics']['test_mse']:.4f}")
    
    # Plot regression results
    regression_fig = plot_regression_results(results)
    regression_fig.show()

if __name__ == "__main__":
    main()