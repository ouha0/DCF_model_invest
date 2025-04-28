# Import packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Union, Optional


class DCFAnalyzer:
    """
    A class for Discounted Cash Flow analysis of companies.
    Calculates intrinsic value ranges based on historical financial data.
    """

    # Initial setup
    def __init__(self, file_path: str):
        """Initialize the DCF analyzer with a data file."""
        self.file_path = file_path
        self.df = None
        self.raw_financials = None
        self.conservative_mode = "average"  # Default level of conservatism
        self.params = {}  # Store calculation parameters
        self.results = {}  # Store analysis results
        
        # Load data
        self.load_data()
        
    
    # Method to load financial data from excel 
    def load_data(self) -> None:
        """Load financial data from Excel file."""
        try:
            self.df = pd.read_excel(self.file_path)
            
            # Create dataframes for different aspects of the data
            self.cash_flow_data = self.df[['Year', 'Cash Flows', 'Capital']].copy()
            self.company_data = self.df[['Company', 'Start', 'End']].dropna(axis=0)
            self.assets_data = self.df[['Year', 'Assets']].fillna(0)
            
            # Basic data checks
            if self.cash_flow_data.empty:
                raise ValueError("Cash flow data is empty")
                
            # Calculate key metrics
            self._calculate_base_metrics()
            print("Data loaded successfully.")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _calculate_base_metrics(self) -> None:
        """Calculate basic metrics from the loaded data."""
        # Cash flow metrics: use minimum cashflow, max cashflow, max cashflow
        self.cf_min = self.df["Cash Flows"].min()
        self.cf_avg = self.df["Cash Flows"].mean()
        self.cf_max = self.df["Cash Flows"].max()
        self.cf_latest = self.df["Cash Flows"].iloc[-1]
        
        # Year range
        self.year_begin = self.df["Year"].min()
        self.year_end = self.df["Year"].max()
        self.num_years_historical = len(self.df["Year"].unique())
        
        # Capital data
        self.cap_avg = self.df["Capital"].mean()
        self.cap_min = self.df["Capital"].min()
        
        # Calculate selling multiples
        self.cash_flow_data["Selling Multiple"] = self.cash_flow_data["Capital"] / self.cash_flow_data["Cash Flows"]
        self.min_selling_mult = self.cash_flow_data["Selling Multiple"].min()
        self.avg_selling_mult = self.cash_flow_data["Selling Multiple"].mean()
        
        # Calculate historical growth rates
        self._calculate_growth_rates()
    
    # Use this?
    def _calculate_growth_rates(self) -> None:
        """Calculate historical growth rates for different metrics."""
        # Calculate cash flow growth rates
        self.cash_flow_data["Growth"] = self.cash_flow_data["Cash Flows"].pct_change() * 100
        
        # Calculate average annual growth rate using CAGR formula
        if len(self.cash_flow_data) > 1:
            first_cf = self.cash_flow_data["Cash Flows"].iloc[0]
            last_cf = self.cash_flow_data["Cash Flows"].iloc[-1]
            years = len(self.cash_flow_data) - 1
            
            if first_cf > 0:  # Prevent math domain errors
                self.cagr = ((last_cf / first_cf) ** (1 / years)) - 1
            else:
                self.cagr = 0
        else:
            self.cagr = 0
            
        self.avg_growth_rate = self.cash_flow_data["Growth"].mean() / 100 if not self.cash_flow_data["Growth"].empty else 0
        
    def set_conservative_mode(self, mode: str) -> None:
        """
        Set the conservative mode for analysis.
        
        Args:
            mode (str): One of 'conservative', 'average', 'not conservative', or 'manual'
        """
        valid_modes = ['conservative', 'average', 'not conservative', 'manual']
        if mode not in valid_modes:
            print(f"Invalid mode. Using default: average")
            self.conservative_mode = "average"
        else:
            self.conservative_mode = mode
            
        # Set default parameters based on conservative mode
        if mode == "conservative":
            self.params["discount_rate"] = 0.20
            self.params["margin_safety"] = 0.50
        elif mode == "average":
            self.params["discount_rate"] = 0.15
            self.params["margin_safety"] = 0.30
        elif mode == "not conservative":
            self.params["discount_rate"] = 0.10
            self.params["margin_safety"] = 0.30
        
        print(f"Mode set to: {self.conservative_mode}")
        
    # Convert to float
    def _validate_numeric_input(self, value: str, default: float) -> float:
        """Validate that input can be converted to a float."""
        try:
            return float(value)
        except ValueError:
            print(f"Invalid input. Using default value: {default}")
            return default
    
    def get_user_inputs(self) -> None:
        """Collect all required inputs from the user for DCF analysis."""
        if self.conservative_mode == "manual":
            # Discount rate
            discount_input = input("What is the discount rate? (%, standard is 15%): ")
            self.params["discount_rate"] = self._validate_numeric_input(discount_input, 15.0) / 100
            
            # Growth rates
            print(f"Historical CAGR: {self.cagr*100:.2f}%")
            min_growth_input = input(f"What is the minimum expected growth rate? (%, suggested: {max(1, self.cagr*50):.1f}%): ")
            self.params["min_growth"] = self._validate_numeric_input(min_growth_input, max(1, self.cagr*50)) / 100
            
            max_growth_input = input(f"What is the maximum expected growth rate? (%, suggested: {max(5, self.cagr*150):.1f}%): ")
            self.params["max_growth"] = self._validate_numeric_input(max_growth_input, max(5, self.cagr*150)) / 100
            
            # Margin of safety
            margin_input = input("What margin of safety would you like to use? (%, standard is 30%): ")
            self.params["margin_safety"] = self._validate_numeric_input(margin_input, 30.0) / 100
            
            # Terminal multiple
            multiple_input = input(f"What terminal multiple should be used? (standard is 10, current avg: {self.avg_selling_mult:.1f}): ")
            self.params["terminal_multiple"] = self._validate_numeric_input(multiple_input, 10.0)
            
            # Forecast years
            years_input = input("How many years forward for prediction? (standard is 10): ")
            self.params["forecast_years"] = int(self._validate_numeric_input(years_input, 10.0))
        else:
            # Use predefined parameters based on conservative mode but still get growth rates
            print(f"Historical CAGR: {self.cagr*100:.2f}%")
            min_growth_input = input(f"What is the minimum expected growth rate? (%, suggested: {max(1, self.cagr*50):.1f}%): ")
            self.params["min_growth"] = self._validate_numeric_input(min_growth_input, max(1, self.cagr*50)) / 100
            
            max_growth_input = input(f"What is the maximum expected growth rate? (%, suggested: {max(5, self.cagr*150):.1f}%): ")
            self.params["max_growth"] = self._validate_numeric_input(max_growth_input, max(5, self.cagr*150)) / 100
            
            # Get terminal multiple
            self.params["terminal_multiple"] = self.avg_selling_mult if not pd.isna(self.avg_selling_mult) else 10.0
            
            # Set forecast years
            self.params["forecast_years"] = 10
    
    def calculate_dcf(self) -> Dict[str, float]:
        """
        Calculate DCF valuation using current parameters.
        
        Returns:
            Dict[str, float]: Dictionary containing intrinsic value ranges
        """
        # Ensure we have all required parameters
        required_params = ["discount_rate", "min_growth", "max_growth", 
                          "margin_safety", "terminal_multiple", "forecast_years"]
        
        if not all(param in self.params for param in required_params):
            print("Missing parameters. Please set parameters first.")
            return {}
        
        # Base cash flow (use latest or minimum)
        base_cf = self.cf_latest  # Using latest is often more realistic
        
        # Calculate lower and upper bound DCF values
        lb_dcf = self._calculate_single_dcf(
            base_cf, 
            self.params["min_growth"], 
            self.params["discount_rate"],
            self.params["terminal_multiple"],
            self.params["forecast_years"]
        )
        
        ub_dcf = self._calculate_single_dcf(
            base_cf, 
            self.params["max_growth"], 
            self.params["discount_rate"],
            self.params["terminal_multiple"],
            self.params["forecast_years"]
        )
        
        # Add assets value if available
        assets_value = self.assets_data["Assets"].sum()
        lb_dcf += assets_value
        ub_dcf += assets_value
        
        # Apply margin of safety
        lb_safety = lb_dcf * (1 - self.params["margin_safety"])
        ub_safety = ub_dcf * (1 - self.params["margin_safety"])
        
        # Store and return results
        self.results = {
            "base_cf": base_cf,
            "lb_intrinsic": lb_dcf,
            "ub_intrinsic": ub_dcf,
            "lb_with_safety": lb_safety,
            "ub_with_safety": ub_safety,
            "assets_value": assets_value,
            "parameters": self.params.copy()
        }
        
        return self.results
    
    def _calculate_single_dcf(self, base_cf: float, growth_rate: float, 
                             discount_rate: float, terminal_multiple: float,
                             forecast_years: int) -> float:
        """
        Calculate a single DCF value based on given parameters.
        
        Args:
            base_cf: Starting cash flow
            growth_rate: Annual growth rate of cash flows
            discount_rate: Discount rate for present value calculations
            terminal_multiple: Multiple for terminal value calculation
            forecast_years: Number of years to forecast
            
        Returns:
            float: DCF valuation
        """
        total_dcf = 0
        yearly_cash_flows = []
        
        # Calculate each year's discounted cash flow
        for year in range(1, forecast_years + 1):
            # Project cash flow for this year
            projected_cf = base_cf * (1 + growth_rate) ** year
            
            # Calculate present value of this cash flow
            present_value = projected_cf / (1 + discount_rate) ** year
            
            # Add to total
            total_dcf += present_value
            
            # Store for potential visualization
            yearly_cash_flows.append({
                'year': self.year_end + year,
                'projected_cf': projected_cf,
                'present_value': present_value
            })
        
        # Calculate terminal value (value of company beyond forecast period)
        terminal_cf = base_cf * (1 + growth_rate) ** forecast_years
        terminal_value = (terminal_cf * terminal_multiple) / (1 + discount_rate) ** forecast_years
        total_dcf += terminal_value
        
        # Store yearly cash flows for visualization
        self.yearly_cash_flows = pd.DataFrame(yearly_cash_flows)
        
        return total_dcf
    
    def display_results(self) -> None:
        """Display DCF analysis results."""
        if not self.results:
            print("No results available. Please run calculate_dcf() first.")
            return
        
        print("\n====== DCF ANALYSIS RESULTS ======")
        print(f"Base Cash Flow: ${self.results['base_cf']:.2f}")
        print(f"Assets Value: ${self.results['assets_value']:.2f}")
        print("\nBefore Margin of Safety:")
        print(f"Lower Bound Intrinsic Value: ${self.results['lb_intrinsic']:.2f}")
        print(f"Upper Bound Intrinsic Value: ${self.results['ub_intrinsic']:.2f}")
        
        print(f"\nAfter {self.params['margin_safety']*100:.0f}% Margin of Safety:")
        print(f"Lower Bound Intrinsic Value: ${self.results['lb_with_safety']:.2f}")
        print(f"Upper Bound Intrinsic Value: ${self.results['ub_with_safety']:.2f}")
        
        print("\nParameters Used:")
        print(f"Discount Rate: {self.params['discount_rate']*100:.1f}%")
        print(f"Growth Rate Range: {self.params['min_growth']*100:.1f}% - {self.params['max_growth']*100:.1f}%")
        print(f"Terminal Multiple: {self.params['terminal_multiple']:.1f}x")
        print(f"Forecast Years: {self.params['forecast_years']}")
        


    def plot_results(self) -> None:
        """Create visualizations of DCF analysis."""
        if not hasattr(self, 'yearly_cash_flows') or self.yearly_cash_flows.empty:
            print("No detailed results available for plotting.")
            return
            
        # Set up the figure
        plt.figure(figsize=(14, 8))
        
        # 1. Plot historical and projected cash flows
        years_historical = self.df['Year'].tolist()
        cf_historical = self.df['Cash Flows'].tolist()
        
        years_projected = self.yearly_cash_flows['year'].tolist()
        cf_projected_min = [self.cf_latest * (1 + self.params['min_growth']) ** (i+1) for i in range(len(years_projected))]
        cf_projected_max = [self.cf_latest * (1 + self.params['max_growth']) ** (i+1) for i in range(len(years_projected))]
        
        plt.subplot(2, 2, 1)
        plt.plot(years_historical, cf_historical, 'b-o', label='Historical')
        plt.plot(years_projected, cf_projected_min, 'g--', label=f"Min Growth ({self.params['min_growth']*100:.1f}%)")
        plt.plot(years_projected, cf_projected_max, 'r--', label=f"Max Growth ({self.params['max_growth']*100:.1f}%)")
        plt.title('Cash Flow Projection')
        plt.xlabel('Year')
        plt.ylabel('Cash Flow')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Present value of projected cash flows
        plt.subplot(2, 2, 2)
        plt.bar(self.yearly_cash_flows['year'], self.yearly_cash_flows['present_value'])
        plt.title('Present Value of Projected Cash Flows')
        plt.xlabel('Year')
        plt.ylabel('Present Value')
        plt.grid(True, alpha=0.3)
        
        # 3. Intrinsic value range with and without margin of safety
        plt.subplot(2, 2, 3)
        values = [self.results['lb_intrinsic'], self.results['ub_intrinsic'], 
                 self.results['lb_with_safety'], self.results['ub_with_safety']]
        labels = ['Lower\nBound', 'Upper\nBound', 'Lower Bound\nw/ Safety', 'Upper Bound\nw/ Safety']
        
        plt.bar(labels, values, color=['blue', 'blue', 'green', 'green'], alpha = 0.7)
        plt.title('Intrinsic Value Range')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        # 4. Sensitivity analysis of discount rate
        plt.subplot(2, 2, 4)
        discount_rates = np.linspace(max(0.05, self.params['discount_rate']-0.05), 
                                    min(0.25, self.params['discount_rate']+0.05), 5)
        
        values_at_rates_min = []
        values_at_rates_max = []
        
        for rate in discount_rates:
            val_min = self._calculate_single_dcf(
                self.results['base_cf'],
                self.params['min_growth'],
                rate,
                self.params['terminal_multiple'],
                self.params['forecast_years']
            )
            val_max = self._calculate_single_dcf(
                self.results['base_cf'],
                self.params['max_growth'],
                rate,
                self.params['terminal_multiple'],
                self.params['forecast_years']
            )
            values_at_rates_min.append(val_min)
            values_at_rates_max.append(val_max)
            
        plt.plot(discount_rates*100, values_at_rates_min, 'g-o', label='Min Growth')  
        plt.plot(discount_rates*100, values_at_rates_max, 'r-o', label='Max Growth')
        plt.title('Sensitivity to Discount Rate')
        plt.xlabel('Discount Rate (%)')
        plt.ylabel('Intrinsic Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_qualitative_analysis(self) -> float:
        """Run qualitative analysis by asking questions about the company."""
        questions = [
            "Does this business have competitive advantages? (1-10): ",
            "How is the management quality? (1-10): ",
            "Is the current price worth it compared to value? (1-10): ",
            "How hyped is the stock? (10=not hyped, 1=very hyped): ",
            "How adaptable is the company to change? (1-10): ",
            "How manageable is the company's debt? (1=high debt, 10=low debt): ",
            "How efficiently is capital allocated? (1-10): ",
            "How loyal are the company's customers? (1-10): ",
            "How protected is the company from technological obsolescence? (1-10): ",
            "How favorable are current business prospects? (1-10): ",
            "Has the company survived market downturns well? (1-10): "
        ]
        
        weights = [1.5, 1.2, 1.0, 0.8, 1.0, 1.0, 1.2, 0.9, 1.0, 0.8, 0.6]  # Weights for each question
        
        print("\n===== QUALITATIVE ANALYSIS =====")
        print("Rate each factor from 1-10 (10 being the best)")
        
        scores = []
        for i, question in enumerate(questions):
            while True:
                try:
                    score = float(input(question))
                    if 1 <= score <= 10:
                        scores.append(score)
                        break
                    else:
                        print("Please enter a number between 1 and 10")
                except ValueError:
                    print("Please enter a valid number")
        
        # Calculate weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        weighted_avg = weighted_sum / total_weight
        
        # Simple average for comparison
        simple_avg = sum(scores) / len(scores)
        
        print(f"\nQualitative Analysis Results:")
        print(f"Simple Average Score: {simple_avg:.2f}/10")
        print(f"Weighted Average Score: {weighted_avg:.2f}/10")
        
        # Interpretation
        if weighted_avg >= 8.5:
            print("Interpretation: Excellent company quality")
        elif weighted_avg >= 7.0:
            print("Interpretation: Good company quality")
        elif weighted_avg >= 5.5:
            print("Interpretation: Average company quality")
        elif weighted_avg >= 4.0:
            print("Interpretation: Below average company quality")
        else:
            print("Interpretation: Poor company quality")
            
        return weighted_avg
    
    def save_results(self, filename: str = 'dcf_analysis_results.xlsx') -> None:
        """Save analysis results to Excel file."""
        if not self.results:
            print("No results available to save.")
            return
            
        # Create a writer object
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        
        # Create summary sheet
        summary_data = {
            'Metric': [
                'Base Cash Flow',
                'Assets Value',
                'Lower Bound Intrinsic Value (Before Safety)',
                'Upper Bound Intrinsic Value (Before Safety)',
                'Lower Bound Intrinsic Value (After Safety)',
                'Upper Bound Intrinsic Value (After Safety)',
                'Discount Rate',
                'Minimum Growth Rate',
                'Maximum Growth Rate',
                'Margin of Safety',
                'Terminal Multiple',
                'Forecast Years'
            ],
            'Value': [
                self.results['base_cf'],
                self.results['assets_value'],
                self.results['lb_intrinsic'],
                self.results['ub_intrinsic'],
                self.results['lb_with_safety'],
                self.results['ub_with_safety'],
                f"{self.params['discount_rate']*100:.1f}%",
                f"{self.params['min_growth']*100:.1f}%",
                f"{self.params['max_growth']*100:.1f}%",
                f"{self.params['margin_safety']*100:.1f}%",
                self.params['terminal_multiple'],
                self.params['forecast_years']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create projected cash flows sheet if available
        if hasattr(self, 'yearly_cash_flows') and not self.yearly_cash_flows.empty:
            self.yearly_cash_flows.to_excel(writer, sheet_name='Projected Cash Flows', index=False)
        
        # Add historical data
        self.df.to_excel(writer, sheet_name='Historical Data', index=False)
        
        # Save the file
        writer.close()
        print(f"Analysis results saved to {filename}")


def main():
    """Main function to run DCF analysis."""
    print("===== DCF ANALYSIS TOOL =====")
    
    # Get file path
    file_path = input("Enter path to Excel data file (default: 'company_data.xlsx'): ") or "company_data.xlsx"
    
    try:
        # Initialize analyzer
        analyzer = DCFAnalyzer(file_path)
        
        # Get conservative mode
        print("\nSelect analysis mode:")
        print("1: Conservative (20% discount rate, 50% margin of safety)")
        print("2: Average (15% discount rate, 30% margin of safety)")
        print("3: Not Conservative (10% discount rate, 30% margin of safety)")
        print("4: Manual (Set parameters manually)")
        
        mode_choice = input("Enter choice (1-4): ")
        if mode_choice == "1":
            analyzer.set_conservative_mode("conservative")
        elif mode_choice == "2":
            analyzer.set_conservative_mode("average")
        elif mode_choice == "3":
            analyzer.set_conservative_mode("not conservative")
        elif mode_choice == "4":
            analyzer.set_conservative_mode("manual")
        else:
            print("Invalid choice. Using 'average' mode.")
            analyzer.set_conservative_mode("average")
        
        # Get user inputs for remaining parameters
        analyzer.get_user_inputs()
        
        # Calculate DCF
        analyzer.calculate_dcf()
        
        # Display results
        analyzer.display_results()
        
        # Visualization
        try:
            analyzer.plot_results()
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        # Ask user if they want to do qualitative analysis
        if input("\nWould you like to perform qualitative analysis? (y/n): ").lower() == 'y':
            analyzer.run_qualitative_analysis()
        
        # Ask user if they want to save results
        if input("\nWould you like to save the analysis results? (y/n): ").lower() == 'y':
            filename = input("Enter filename (default: dcf_analysis_results.xlsx): ") or "dcf_analysis_results.xlsx"
            analyzer.save_results(filename)
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()


