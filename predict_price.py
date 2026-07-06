# ============================================================
# CAR PRICE PREDICTION - DEPLOYMENT SCRIPT
# ============================================================
# الهدف: تحميل النموذج وتنفيذ التنبؤات على سيارات جديدة
# الاستخدام: python predict_price.py
# أو استدعاء دالة predict_price() من كود آخر
# ============================================================

import joblib
import pandas as pd
import numpy as np
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    """
    Class for loading the trained model and predicting car prices
    """
    
    def __init__(self, model_path='models/best_model.pkl', 
                 encoders_path='models/label_encoders.pkl',
                 features_path='models/features.pkl',
                 scaler_path=None):  # ✅ Default to None - Gradient Boosting doesn't need scaler
        """
        Initialize the predictor by loading all artifacts
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file (Gradient Boosting)
        encoders_path : str
            Path to the label encoders file
        features_path : str
            Path to the features list file
        scaler_path : str, optional
            Path to the scaler file (not needed for Gradient Boosting)
        """
        print("📦 Loading model artifacts...")
        
        # Load model
        try:
            self.model = joblib.load(model_path)
            print(f"  ✅ Model loaded from {model_path}")
            print(f"  📊 Model type: {type(self.model).__name__}")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            raise
        
        # Load label encoders
        try:
            self.label_encoders = joblib.load(encoders_path)
            print(f"  ✅ Encoders loaded from {encoders_path}")
            for key, encoder in self.label_encoders.items():
                print(f"     - {key}: {len(encoder.classes_)} categories")
        except Exception as e:
            print(f"  ❌ Error loading encoders: {e}")
            raise
        
        # Load features
        try:
            self.features = joblib.load(features_path)
            print(f"  ✅ Features loaded from {features_path}")
            print(f"     - {len(self.features)} features: {self.features}")
        except Exception as e:
            print(f"  ❌ Error loading features: {e}")
            raise
        
        # Load scaler if provided (optional - for linear models only)
        self.scaler = None
        if scaler_path:
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"  ✅ Scaler loaded from {scaler_path}")
                print(f"  ⚠️  Note: Gradient Boosting does NOT require scaling!")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not load scaler: {e}")
                self.scaler = None
        else:
            print(f"  ℹ️  No scaler loaded (not needed for Gradient Boosting)")
        
        # Verify model type
        if 'GradientBoosting' in str(type(self.model)):
            print("  ✅ Model verified: Gradient Boosting (no scaling needed)")
        else:
            print(f"  ⚠️  Warning: Model is {type(self.model).__name__}, not Gradient Boosting")
            print(f"     If using a linear model, you may need scaling!")
        
        print("✅ Predictor ready!\n")
    
    def validate_input(self, car_info):
        """
        Validate that all required fields are present and have correct types
        
        Parameters:
        -----------
        car_info : dict
            Dictionary containing car information
            
        Returns:
        --------
        tuple (bool, str) - (is_valid, error_message)
        """
        required_fields = [
            'brand', 'vehicle_age', 'km_driven', 'fuel_type',
            'transmission_type', 'seller_type', 'engine', 'max_power', 'seats'
        ]
        
        # Check for missing fields
        missing = [field for field in required_fields if field not in car_info]
        if missing:
            return False, f"Missing fields: {missing}"
        
        # Validate data types and ranges
        try:
            if not isinstance(car_info['brand'], str):
                return False, "brand must be a string"
            
            if not isinstance(car_info['vehicle_age'], (int, float)) or car_info['vehicle_age'] < 0 or car_info['vehicle_age'] > 30:
                return False, "vehicle_age must be between 0 and 30"
            
            if not isinstance(car_info['km_driven'], (int, float)) or car_info['km_driven'] < 0:
                return False, "km_driven must be positive"
            
            if not isinstance(car_info['engine'], (int, float)) or car_info['engine'] < 500 or car_info['engine'] > 6000:
                return False, "engine must be between 500 and 6000 cc"
            
            if not isinstance(car_info['max_power'], (int, float)) or car_info['max_power'] < 30 or car_info['max_power'] > 400:
                return False, "max_power must be between 30 and 400 BHP"
            
            if not isinstance(car_info['seats'], int) or car_info['seats'] not in [2, 4, 5, 6, 7, 8, 9]:
                return False, "seats must be 2,4,5,6,7,8,9"
            
            # Check categorical values exist in encoders
            if car_info['brand'] not in self.label_encoders['brand'].classes_:
                similar_brands = [b for b in self.label_encoders['brand'].classes_ 
                                 if car_info['brand'].lower() in b.lower()]
                if similar_brands:
                    return False, f"Brand '{car_info['brand']}' not found. Did you mean: {similar_brands[:3]}?"
                return False, f"Brand '{car_info['brand']}' not in trained brands. Available: {list(self.label_encoders['brand'].classes_)[:10]}..."
            
            if car_info['fuel_type'] not in self.label_encoders['fuel_type'].classes_:
                return False, f"Fuel type '{car_info['fuel_type']}' not found. Available: {list(self.label_encoders['fuel_type'].classes_)}"
            
            if car_info['transmission_type'] not in self.label_encoders['transmission_type'].classes_:
                return False, f"Transmission '{car_info['transmission_type']}' not found. Available: {list(self.label_encoders['transmission_type'].classes_)}"
            
            if car_info['seller_type'] not in self.label_encoders['seller_type'].classes_:
                return False, f"Seller type '{car_info['seller_type']}' not found. Available: {list(self.label_encoders['seller_type'].classes_)}"
            
        except Exception as e:
            return False, f"Validation error: {e}"
        
        return True, "OK"
    
    def predict(self, car_info, return_details=False):
        """
        Predict car price using Gradient Boosting (no scaling required)
        
        Parameters:
        -----------
        car_info : dict
            Dictionary containing car information with keys:
            - brand: str
            - vehicle_age: int (years)
            - km_driven: int (kilometers)
            - fuel_type: str (Petrol/Diesel/CNG/LPG)
            - transmission_type: str (Manual/Automatic)
            - seller_type: str (Dealer/Individual/Trustmark Dealer)
            - engine: int (cc)
            - max_power: float (BHP)
            - seats: int (2-9)
        
        return_details : bool
            If True, returns dict with prediction and additional info
        
        Returns:
        --------
        float or dict: Predicted price in EGP
        """
        # Validate input
        is_valid, error_msg = self.validate_input(car_info)
        if not is_valid:
            raise ValueError(f"Invalid input: {error_msg}")
        
        # Encode categorical variables
        try:
            brand_enc = self.label_encoders['brand'].transform([car_info['brand']])[0]
            fuel_enc = self.label_encoders['fuel_type'].transform([car_info['fuel_type']])[0]
            trans_enc = self.label_encoders['transmission_type'].transform([car_info['transmission_type']])[0]
            seller_enc = self.label_encoders['seller_type'].transform([car_info['seller_type']])[0]
        except Exception as e:
            raise ValueError(f"Encoding error: {e}")
        
        # Create feature array (order must match training)
        input_features = [
            car_info['vehicle_age'],
            car_info['km_driven'],
            car_info['engine'],
            car_info['max_power'],
            car_info['seats'],
            brand_enc,
            fuel_enc,
            trans_enc,
            seller_enc
        ]
        
        # ❌ NO SCALING - Gradient Boosting is tree-based and doesn't need it!
        # ⚠️  If you're using a linear model (LinearRegression, Ridge, Lasso),
        #     you would need to apply scaling here.
        #
        # if self.scaler:  # Only for linear models
        #     input_features = self.scaler.transform([input_features])[0]
        
        # Predict directly with Gradient Boosting
        try:
            prediction = self.model.predict([input_features])[0]
        except Exception as e:
            raise ValueError(f"Prediction error: {e}")
        
        # Ensure minimum reasonable price
        prediction = max(prediction, 50000)
        
        if return_details:
            return {
                'price': prediction,
                'price_formatted': f"{prediction:,.0f} EGP",
                'features_used': self.features,
                'input_features': input_features,
                'encoded_values': {
                    'brand_encoded': brand_enc,
                    'fuel_encoded': fuel_enc,
                    'transmission_encoded': trans_enc,
                    'seller_encoded': seller_enc
                },
                'model_type': type(self.model).__name__
            }
        
        return prediction
    
    def predict_batch(self, cars_list):
        """
        Predict prices for multiple cars
        
        Parameters:
        -----------
        cars_list : list of dict
            List of car information dictionaries
            
        Returns:
        --------
        list: List of prediction results
        """
        predictions = []
        errors = []
        
        for i, car in enumerate(cars_list):
            try:
                price = self.predict(car)
                predictions.append({
                    'car': car, 
                    'price': price, 
                    'price_formatted': f"{price:,.0f} EGP",
                    'error': None
                })
            except Exception as e:
                predictions.append({
                    'car': car, 
                    'price': None, 
                    'price_formatted': None,
                    'error': str(e)
                })
                errors.append((i, str(e)))
        
        if errors:
            print(f"⚠️ {len(errors)} predictions failed")
        
        return predictions


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def interactive_test():
    """
    Run interactive test with user input
    """
    print("\n" + "="*60)
    print("🔧    CAR PRICE PREDICTOR - INTERACTIVE MODE")
    print("="*60)
    print("Enter car details to get price prediction")
    print("(Press Ctrl+C to exit)\n")
    
    predictor = CarPricePredictor()
    
    # Available options
    available_brands = predictor.label_encoders['brand'].classes_.tolist()
    available_fuels = predictor.label_encoders['fuel_type'].classes_.tolist()
    available_trans = predictor.label_encoders['transmission_type'].classes_.tolist()
    available_sellers = predictor.label_encoders['seller_type'].classes_.tolist()
    
    print(f"📋 Available brands: {', '.join(available_brands[:15])}...")
    print(f"📋 Available fuel types: {available_fuels}")
    print(f"📋 Available transmissions: {available_trans}")
    print(f"📋 Available seller types: {available_sellers}")
    print("-"*60)
    
    while True:
        try:
            print("\n" + "─"*50)
            
            # Get user input with validation loop
            while True:
                brand = input("Brand: ").strip()
                # Case-insensitive matching
                matching_brands = [b for b in available_brands if b.lower() == brand.lower()]
                if matching_brands:
                    brand = matching_brands[0]
                    break
                elif brand.lower() in [b.lower() for b in available_brands]:
                    brand = [b for b in available_brands if b.lower() == brand.lower()][0]
                    break
                else:
                    print(f"❌ Brand not found. Available: {available_brands[:10]}...")
            
            age = int(input("Vehicle age (years): "))
            km = int(input("Kilometers driven: "))
            
            while True:
                fuel = input(f"Fuel type {available_fuels}: ").strip()
                if fuel in available_fuels:
                    break
                print(f"❌ Invalid. Choose from: {available_fuels}")
            
            while True:
                trans = input(f"Transmission {available_trans}: ").strip()
                if trans in available_trans:
                    break
                print(f"❌ Invalid. Choose from: {available_trans}")
            
            while True:
                seller = input(f"Seller type {available_sellers}: ").strip()
                if seller in available_sellers:
                    break
                print(f"❌ Invalid. Choose from: {available_sellers}")
            
            engine = int(input("Engine size (cc): "))
            power = float(input("Max power (BHP): "))
            seats = int(input("Number of seats: "))
            
            car = {
                'brand': brand,
                'vehicle_age': age,
                'km_driven': km,
                'fuel_type': fuel,
                'transmission_type': trans,
                'seller_type': seller,
                'engine': engine,
                'max_power': power,
                'seats': seats
            }
            
            result = predictor.predict(car, return_details=True)
            print("\n" + "="*50)
            print(f"💰 PREDICTED PRICE: {result['price_formatted']}")
            print(f"📊 Model: {result['model_type']}")
            print("="*50)
            
            again = input("\nPredict another car? (y/n): ").strip().lower()
            if again != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with valid inputs.\n")


def quick_test():
    """
    Run quick test with predefined examples
    """
    print("\n" + "="*60)
    print("🚗 QUICK TEST - PREDEFINED EXAMPLES")
    print("="*60)
    
    predictor = CarPricePredictor()
    
    test_cases = [
        {
            'name': 'Economy Car - Hyundai i10',
            'car': {
                'brand': 'Hyundai',
                'vehicle_age': 3,
                'km_driven': 40000,
                'fuel_type': 'Petrol',
                'transmission_type': 'Manual',
                'seller_type': 'Individual',
                'engine': 1197,
                'max_power': 82,
                'seats': 5
            }
        },
        {
            'name': 'Family Car - Toyota Innova',
            'car': {
                'brand': 'Toyota',
                'vehicle_age': 5,
                'km_driven': 80000,
                'fuel_type': 'Diesel',
                'transmission_type': 'Manual',
                'seller_type': 'Dealer',
                'engine': 1995,
                'max_power': 140,
                'seats': 7
            }
        },
        {
            'name': 'Luxury Car - Mercedes-Benz',
            'car': {
                'brand': 'Mercedes-Benz',
                'vehicle_age': 2,
                'km_driven': 25000,
                'fuel_type': 'Petrol',
                'transmission_type': 'Automatic',
                'seller_type': 'Dealer',
                'engine': 2000,
                'max_power': 180,
                'seats': 5
            }
        }
    ]
    
    print("\n📊 Results:\n")
    for test in test_cases:
        try:
            result = predictor.predict(test['car'], return_details=True)
            print(f"  {test['name']}:")
            print(f"    → {result['price_formatted']}")
            print(f"    → Model: {result['model_type']}\n")
        except Exception as e:
            print(f"  {test['name']}: ❌ Error - {e}\n")


def batch_test_from_json(json_file):
    """
    Run batch prediction from JSON file
    
    Parameters:
    -----------
    json_file : str
        Path to JSON file containing list of car dictionaries
    """
    print("\n" + "="*60)
    print("📦 BATCH PREDICTION")
    print("="*60)
    
    try:
        with open(json_file, 'r') as f:
            cars = json.load(f)
        print(f"✅ Loaded {len(cars)} cars from {json_file}")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return
    
    predictor = CarPricePredictor()
    results = predictor.predict_batch(cars)
    
    print("\n📊 Results:\n")
    print("-"*60)
    
    success_count = sum(1 for r in results if r['error'] is None)
    print(f"✅ Successful: {success_count}/{len(results)}")
    print("-"*60)
    
    for i, result in enumerate(results, 1):
        if result['error']:
            print(f"{i}. ❌ {result['car'].get('brand', 'Unknown')}: {result['error']}")
        else:
            print(f"{i}. ✅ {result['car']['brand']} {result['car'].get('model', '')}: {result['price_formatted']}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Car Price Prediction Tool - Gradient Boosting Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_price.py --mode quick
  python predict_price.py --mode interactive
  python predict_price.py --mode batch --input cars.json
        """
    )
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'quick', 'batch'],
                        help='Run mode: interactive, quick, or batch')
    parser.add_argument('--input', type=str, 
                        help='JSON file for batch prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_test()
    elif args.mode == 'interactive':
        interactive_test()
    elif args.mode == 'batch':
        if args.input:
            batch_test_from_json(args.input)
        else:
            print("❌ Please provide --input JSON file for batch mode")
            print("Example: python predict_price.py --mode batch --input cars.json")
    else:
        parser.print_help()