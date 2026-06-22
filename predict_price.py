# ============================================================
# CAR PRICE PREDICTION - DEPLOYMENT SCRIPT
# ============================================================

import joblib
import pandas as pd
import numpy as np
import sys
import json
import argparse

class CarPricePredictor:
    """
    Class for loading the trained model and predicting car prices
    """
    
    def __init__(self, model_path='models/best_model.pkl', 
                 encoders_path='models/label_encoders.pkl',
                 features_path='models/features.pkl',
                 scaler_path='models/scaler.pkl'):
        """
        Initialize the predictor by loading all artifacts
        """
        print("📦 Loading model artifacts...")
        
        try:
            self.model = joblib.load(model_path)
            print(f"  ✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            raise
        
        try:
            self.label_encoders = joblib.load(encoders_path)
            print(f"  ✅ Encoders loaded from {encoders_path}")
        except Exception as e:
            print(f"  ❌ Error loading encoders: {e}")
            raise
        
        try:
            self.features = joblib.load(features_path)
            print(f"  ✅ Features loaded from {features_path}")
        except Exception as e:
            print(f"  ❌ Error loading features: {e}")
            raise
        
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"  ✅ Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not load scaler: {e}")
            self.scaler = None
        
        print("✅ Predictor ready!\n")
    
    def validate_input(self, car_info):
        """Validate that all required fields are present"""
        required_fields = [
            'brand', 'vehicle_age', 'km_driven', 'fuel_type',
            'transmission_type', 'seller_type', 'engine', 'max_power', 'seats'
        ]
        
        missing = [field for field in required_fields if field not in car_info]
        if missing:
            return False, f"Missing fields: {missing}"
        
        # Check categorical values
        try:
            if car_info['brand'] not in self.label_encoders['brand'].classes_:
                return False, f"Brand '{car_info['brand']}' not found"
            
            if car_info['fuel_type'] not in self.label_encoders['fuel_type'].classes_:
                return False, f"Fuel type '{car_info['fuel_type']}' not found"
            
            if car_info['transmission_type'] not in self.label_encoders['transmission_type'].classes_:
                return False, f"Transmission '{car_info['transmission_type']}' not found"
            
            if car_info['seller_type'] not in self.label_encoders['seller_type'].classes_:
                return False, f"Seller type '{car_info['seller_type']}' not found"
            
        except Exception as e:
            return False, f"Validation error: {e}"
        
        return True, "OK"
    
    def predict(self, car_info, return_details=False):
        """Predict car price"""
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
        
        # Create feature array
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
        
        # Apply scaler
        if self.scaler:
            input_features = self.scaler.transform([input_features])[0]
        
        # Predict
        try:
            prediction = self.model.predict([input_features])[0]
        except Exception as e:
            raise ValueError(f"Prediction error: {e}")
        
        prediction = max(prediction, 50000)
        
        if return_details:
            return {
                'price': prediction,
                'price_formatted': f"{prediction:,.0f} EGP",
                'features_used': self.features,
                'encoded_values': {
                    'brand_encoded': brand_enc,
                    'fuel_encoded': fuel_enc,
                    'transmission_encoded': trans_enc,
                    'seller_encoded': seller_enc
                }
            }
        
        return prediction
    
    def predict_batch(self, cars_list):
        """Predict prices for multiple cars"""
        predictions = []
        for car in cars_list:
            try:
                price = self.predict(car)
                predictions.append({'car': car, 'price': price, 'error': None})
            except Exception as e:
                predictions.append({'car': car, 'price': None, 'error': str(e)})
        return predictions


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def interactive_test():
    """Run interactive test with user input"""
    print("\n" + "="*60)
    print("🔧 CAR PRICE PREDICTOR - INTERACTIVE MODE")
    print("="*60)
    
    predictor = CarPricePredictor()
    
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
            
            brand = input("Brand: ").strip()
            matching_brands = [b for b in available_brands if b.lower() == brand.lower()]
            if matching_brands:
                brand = matching_brands[0]
            else:
                print(f"❌ Brand not found. Available: {available_brands[:10]}...")
                continue
            
            age = int(input("Vehicle age (years): "))
            km = int(input("Kilometers driven: "))
            
            fuel = input(f"Fuel type {available_fuels}: ").strip()
            if fuel not in available_fuels:
                print(f"❌ Invalid. Choose from: {available_fuels}")
                continue
            
            trans = input(f"Transmission {available_trans}: ").strip()
            if trans not in available_trans:
                print(f"❌ Invalid. Choose from: {available_trans}")
                continue
            
            seller = input(f"Seller type {available_sellers}: ").strip()
            if seller not in available_sellers:
                print(f"❌ Invalid. Choose from: {available_sellers}")
                continue
            
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
    """Run quick test with predefined examples"""
    print("\n" + "="*60)
    print("🚗 QUICK TEST - PREDEFINED EXAMPLES")
    print("="*60)
    
    predictor = CarPricePredictor()
    
    test_cases = [
        {
            'name': 'Economy Car - Hyundai',
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
            'name': 'Family Car - Toyota',
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
            price = predictor.predict(test['car'])
            print(f"  {test['name']}:")
            print(f"    → {price:,.0f} EGP\n")
        except Exception as e:
            print(f"  {test['name']}: ❌ Error - {e}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Car Price Prediction Tool')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'quick', 'batch'],
                        help='Run mode: interactive, quick, or batch')
    parser.add_argument('--input', type=str, help='JSON file for batch prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_test()
    elif args.mode == 'interactive':
        interactive_test()
    elif args.mode == 'batch':
        if args.input:
            with open(args.input, 'r') as f:
                cars = json.load(f)
            predictor = CarPricePredictor()
            results = predictor.predict_batch(cars)
            for r in results:
                if r['error']:
                    print(f"❌ {r['car'].get('brand', 'Unknown')}: {r['error']}")
                else:
                    print(f"✅ {r['car']['brand']}: {r['price']:,.0f} EGP")
        else:
            print("Please provide --input JSON file for batch mode")