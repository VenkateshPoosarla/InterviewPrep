"""
Sample Data Generator for Propensity Modeling

Generates realistic synthetic customer data with conversion labels
for training propensity models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


class CustomerDataGenerator:
    """Generate synthetic customer data with realistic patterns"""

    def __init__(self, n_customers=100000):
        self.n_customers = n_customers

    def generate(self):
        """Generate complete customer dataset"""
        print(f"Generating data for {self.n_customers:,} customers...")

        # Generate customer IDs
        customer_ids = [f"CUST_{i:07d}" for i in range(self.n_customers)]

        # Demographics
        demographics = self._generate_demographics()

        # Behavioral features
        behavioral = self._generate_behavioral()

        # Engagement features
        engagement = self._generate_engagement()

        # Transaction features
        transactions = self._generate_transactions()

        # Combine all features
        data = pd.DataFrame({
            'customer_id': customer_ids,
            **demographics,
            **behavioral,
            **engagement,
            **transactions
        })

        # Generate conversion label (target variable)
        data['converted'] = self._generate_conversion_labels(data)

        print(f"✓ Generated {len(data):,} customer records")
        print(f"  - Features: {len(data.columns) - 2}")
        print(f"  - Conversion rate: {data['converted'].mean():.2%}")

        return data

    def _generate_demographics(self):
        """Generate demographic features"""
        return {
            'age': np.random.normal(40, 15, self.n_customers).clip(18, 80).astype(int),
            'gender': np.random.choice(['M', 'F', 'Other'], self.n_customers, p=[0.48, 0.48, 0.04]),
            'income_bracket': np.random.choice(
                ['<30K', '30-50K', '50-75K', '75-100K', '100K+'],
                self.n_customers,
                p=[0.15, 0.25, 0.30, 0.20, 0.10]
            ),
            'education': np.random.choice(
                ['High School', 'Some College', 'Bachelor', 'Graduate'],
                self.n_customers,
                p=[0.20, 0.30, 0.35, 0.15]
            ),
            'location_type': np.random.choice(
                ['Urban', 'Suburban', 'Rural'],
                self.n_customers,
                p=[0.40, 0.45, 0.15]
            )
        }

    def _generate_behavioral(self):
        """Generate behavioral features"""
        # Customer tenure
        tenure_months = np.random.exponential(24, self.n_customers).clip(0, 120).astype(int)

        # Purchase history
        total_purchases = np.random.poisson(10, self.n_customers)

        # Average order value (AOV)
        aov = np.random.lognormal(4, 0.8, self.n_customers).clip(10, 500)

        # Days since last purchase
        days_since_last = np.random.exponential(30, self.n_customers).clip(0, 365).astype(int)

        # Purchase frequency (purchases per month)
        purchase_frequency = np.where(
            tenure_months > 0,
            total_purchases / tenure_months,
            0
        )

        return {
            'tenure_months': tenure_months,
            'total_purchases': total_purchases,
            'avg_order_value': aov,
            'total_revenue': total_purchases * aov,
            'days_since_last_purchase': days_since_last,
            'purchase_frequency': purchase_frequency,
            'is_repeat_customer': (total_purchases > 1).astype(int)
        }

    def _generate_engagement(self):
        """Generate engagement features"""
        # Email engagement
        email_sent = np.random.poisson(20, self.n_customers)
        email_opened = (email_sent * np.random.beta(2, 3, self.n_customers)).astype(int)
        email_clicked = (email_opened * np.random.beta(1.5, 4, self.n_customers)).astype(int)

        # Website engagement
        website_visits_30d = np.random.poisson(8, self.n_customers)
        avg_session_duration = np.random.gamma(2, 3, self.n_customers).clip(1, 30)
        pages_per_session = np.random.gamma(2, 2, self.n_customers).clip(1, 20)

        # App usage (some customers don't have app)
        has_app = np.random.binomial(1, 0.3, self.n_customers)
        app_sessions_30d = has_app * np.random.poisson(15, self.n_customers)

        # Social media
        social_media_follower = np.random.binomial(1, 0.15, self.n_customers)

        # Calculate rates
        email_open_rate = np.where(email_sent > 0, email_opened / email_sent, 0)
        email_click_rate = np.where(email_opened > 0, email_clicked / email_opened, 0)

        return {
            'email_sent_30d': email_sent,
            'email_opened_30d': email_opened,
            'email_clicked_30d': email_clicked,
            'email_open_rate': email_open_rate,
            'email_click_rate': email_click_rate,
            'website_visits_30d': website_visits_30d,
            'avg_session_duration_min': avg_session_duration,
            'pages_per_session': pages_per_session,
            'has_mobile_app': has_app,
            'app_sessions_30d': app_sessions_30d,
            'social_media_follower': social_media_follower
        }

    def _generate_transactions(self):
        """Generate transaction-based features"""
        # Product categories purchased
        num_categories = np.random.poisson(2, self.n_customers).clip(0, 10)

        # Discount sensitivity
        purchases_with_discount = np.random.beta(2, 3, self.n_customers)

        # Cart abandonment
        cart_abandonment_rate = np.random.beta(2, 2, self.n_customers)

        # Product views
        product_views_30d = np.random.poisson(25, self.n_customers)

        # Wishlist items
        wishlist_items = np.random.poisson(3, self.n_customers)

        # Returns
        return_rate = np.random.beta(1, 10, self.n_customers).clip(0, 0.5)

        return {
            'num_categories_purchased': num_categories,
            'purchases_with_discount_pct': purchases_with_discount,
            'cart_abandonment_rate': cart_abandonment_rate,
            'product_views_30d': product_views_30d,
            'wishlist_items': wishlist_items,
            'return_rate': return_rate
        }

    def _generate_conversion_labels(self, data):
        """
        Generate conversion labels based on features

        High propensity customers:
        - High engagement (email open rate, website visits)
        - Recent activity (low days_since_last_purchase)
        - Good purchase history
        - High AOV
        """
        # Create propensity score based on features
        propensity_score = (
            # Engagement factors (40% weight)
            0.2 * data['email_open_rate'] +
            0.1 * (data['website_visits_30d'] / data['website_visits_30d'].max()) +
            0.1 * (data['app_sessions_30d'] / (data['app_sessions_30d'].max() + 1)) +

            # Recency factor (30% weight)
            0.3 * (1 - data['days_since_last_purchase'] / 365) +

            # Purchase history (20% weight)
            0.1 * (data['total_purchases'] / data['total_purchases'].max()) +
            0.1 * (data['purchase_frequency'] / (data['purchase_frequency'].max() + 1)) +

            # Value factor (10% weight)
            0.1 * (data['avg_order_value'] / data['avg_order_value'].max())
        )

        # Normalize to 0-1
        propensity_score = (propensity_score - propensity_score.min()) / \
                          (propensity_score.max() - propensity_score.min())

        # Add some randomness
        noise = np.random.normal(0, 0.1, len(propensity_score))
        propensity_score = (propensity_score + noise).clip(0, 1)

        # Convert to binary labels
        # Target ~5% conversion rate
        threshold = np.percentile(propensity_score, 95)
        converted = (propensity_score >= threshold).astype(int)

        return converted

    def save(self, data, filename='customer_data.parquet'):
        """Save data to file"""
        filepath = f'data/raw/{filename}'
        data.to_parquet(filepath, index=False)
        print(f"\n✓ Saved data to {filepath}")
        print(f"  File size: {pd.io.common.get_filepath_or_buffer(filepath)[0]}")
        return filepath


def generate_time_series_data(n_customers=10000, n_days=90):
    """
    Generate time-series data for tracking customer behavior over time

    Useful for:
    - Trend analysis
    - Seasonal patterns
    - Recency-Frequency-Monetary (RFM) analysis
    """
    print(f"\nGenerating time-series data...")
    print(f"  Customers: {n_customers:,}")
    print(f"  Time period: {n_days} days")

    records = []
    base_date = datetime.now() - timedelta(days=n_days)

    for customer_id in range(n_customers):
        # Number of events for this customer
        n_events = np.random.poisson(10)

        for _ in range(n_events):
            event_date = base_date + timedelta(
                days=np.random.randint(0, n_days)
            )

            records.append({
                'customer_id': f"CUST_{customer_id:07d}",
                'event_date': event_date,
                'event_type': np.random.choice(
                    ['page_view', 'product_view', 'add_to_cart', 'purchase'],
                    p=[0.60, 0.25, 0.10, 0.05]
                ),
                'event_value': np.random.lognormal(3, 1) if np.random.random() > 0.5 else 0
            })

    df = pd.DataFrame(records)
    df = df.sort_values(['customer_id', 'event_date'])

    filepath = 'data/raw/customer_events.parquet'
    df.to_parquet(filepath, index=False)

    print(f"✓ Generated {len(df):,} events")
    print(f"✓ Saved to {filepath}")

    return df


def main():
    """Generate all sample datasets"""
    print("=" * 60)
    print("PROPENSITY MODEL - SAMPLE DATA GENERATION")
    print("=" * 60)

    # Generate main customer data
    generator = CustomerDataGenerator(n_customers=100000)
    customer_data = generator.generate()
    generator.save(customer_data)

    # Generate time-series data
    events_data = generate_time_series_data(n_customers=10000, n_days=90)

    # Generate summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print("\nCustomer Data:")
    print(customer_data.describe())

    print("\nFeature Correlation with Conversion:")
    numeric_cols = customer_data.select_dtypes(include=[np.number]).columns
    correlations = customer_data[numeric_cols].corr()['converted'].sort_values(ascending=False)
    print(correlations.head(10))

    print("\nConversion Rate by Segment:")
    print(f"  Overall: {customer_data['converted'].mean():.2%}")
    print(f"  Repeat Customers: {customer_data[customer_data['is_repeat_customer']==1]['converted'].mean():.2%}")
    print(f"  New Customers: {customer_data[customer_data['is_repeat_customer']==0]['converted'].mean():.2%}")
    print(f"  App Users: {customer_data[customer_data['has_mobile_app']==1]['converted'].mean():.2%}")
    print(f"  Non-App Users: {customer_data[customer_data['has_mobile_app']==0]['converted'].mean():.2%}")

    print("\n" + "=" * 60)
    print("✓ Sample data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
