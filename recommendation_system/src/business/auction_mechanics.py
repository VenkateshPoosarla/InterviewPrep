"""
Ad Auction Mechanics for Production Ad Ranking System

This module implements auction algorithms and business logic for ad ranking at scale.
Designed for platforms serving billions of users (Roblox, Meta, Google scale).

Key Features:
- Second-price (Vickrey) auction for truthful bidding
- eCPM optimization (expected Cost Per Mille)
- Budget pacing algorithms for smooth spend
- Frequency capping to prevent ad fatigue
- Multi-objective optimization (user experience + revenue)

Core Concepts:
    pCTR: Predicted Click-Through Rate (from ML model)
    bid: Advertiser's bid amount ($ per click)
    eCPM: Expected revenue per 1000 impressions = pCTR * bid * 1000
    Quality Score: Ad relevance/quality metric (0-10 scale)

Auction Design Decisions:
    1. Second-price vs First-price:
       - Second-price: Winner pays 2nd highest bid + ε
       - Advantage: Truthful bidding (no bid shading), stable revenue
       - Used by: Google Ads, Meta Ads

    2. Ranking Score:
       - Traditional: eCPM = pCTR * bid * 1000
       - With quality: eCPM = pCTR * bid * quality_score * 1000
       - Balances revenue with user experience

    3. Multi-Objective:
       - Pure revenue → spammy ads, user churn
       - Pure engagement → low revenue
       - Balance: α*pCTR + β*pCVR + γ*eCPM

Author: Staff ML Engineer Portfolio
Target: Roblox Ad Ranking Role
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class BiddingStrategy(Enum):
    """Advertiser bidding strategies"""
    CPC = "cost_per_click"        # Pay per click
    CPM = "cost_per_mille"        # Pay per 1000 impressions
    CPA = "cost_per_action"       # Pay per conversion
    TARGET_ROAS = "target_roas"   # Target return on ad spend


@dataclass
class Ad:
    """
    Ad candidate with all metadata for auction.

    Attributes:
        ad_id: Unique ad identifier
        advertiser_id: Advertiser account ID
        campaign_id: Campaign identifier
        creative_text: Ad creative (title + description)
        bid_amount: Advertiser's bid ($ per click for CPC)
        bidding_strategy: How advertiser wants to pay
        daily_budget: Daily budget limit ($)
        total_budget: Total campaign budget ($)
        budget_spent_today: Amount spent today ($)
        budget_spent_total: Total amount spent ($)
        quality_score: Ad quality/relevance (0-10)
        predicted_ctr: ML model's CTR prediction (0-1)
        predicted_cvr: ML model's conversion rate prediction (0-1)
        impressions_today: Impressions delivered today
        clicks_today: Clicks received today
        conversions_today: Conversions today
        category: Ad category (gaming, retail, etc.)
        targeting: Targeting criteria
    """
    ad_id: str
    advertiser_id: str
    campaign_id: str
    creative_text: str
    bid_amount: float
    bidding_strategy: BiddingStrategy
    daily_budget: float
    total_budget: float
    budget_spent_today: float
    budget_spent_total: float
    quality_score: float
    predicted_ctr: float
    predicted_cvr: float = 0.0
    impressions_today: int = 0
    clicks_today: int = 0
    conversions_today: int = 0
    category: str = "general"
    targeting: Optional[Dict] = None


@dataclass
class AuctionResult:
    """
    Result of running auction for a single request.

    Attributes:
        winning_ad: The winning ad
        second_price: Price winner will pay
        ecpm: Effective CPM for the winning ad
        rank_score: Final ranking score used
        all_candidates: All ads that participated in auction
    """
    winning_ad: Ad
    second_price: float
    ecpm: float
    rank_score: float
    all_candidates: List[Tuple[Ad, float]]  # (ad, score) sorted by rank


class AdAuctionEngine:
    """
    Production ad auction engine with second-price (Vickrey) auction.

    This implements the core auction logic used in real-time ad serving.
    Runs thousands of times per second, so optimized for low latency.

    Key Features:
    - Second-price auction for truthful bidding
    - eCPM-based ranking (balances relevance and revenue)
    - Budget pacing to prevent budget exhaustion
    - Frequency capping to prevent ad fatigue
    - Quality score for user experience

    Example:
        >>> engine = AdAuctionEngine(
        ...     alpha=0.4,  # Weight for CTR (user engagement)
        ...     beta=0.3,   # Weight for CVR (conversion)
        ...     gamma=0.3   # Weight for revenue
        ... )
        >>> ads = [ad1, ad2, ad3]  # Candidate ads
        >>> result = engine.run_auction(ads, user_id=123)
        >>> print(f"Winner: {result.winning_ad.ad_id}, Price: ${result.second_price:.4f}")
    """

    def __init__(
        self,
        alpha: float = 0.4,  # Weight for user engagement (CTR)
        beta: float = 0.3,   # Weight for conversion (CVR)
        gamma: float = 0.3,  # Weight for revenue (eCPM)
        min_quality_score: float = 3.0,  # Minimum quality to show ad
        budget_pacing_enabled: bool = True,
        frequency_cap_enabled: bool = True
    ):
        """
        Initialize auction engine.

        Args:
            alpha: Weight for CTR in multi-objective optimization
            beta: Weight for CVR
            gamma: Weight for revenue
            min_quality_score: Minimum ad quality to participate
            budget_pacing_enabled: Enable budget pacing
            frequency_cap_enabled: Enable frequency capping
        """
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_quality_score = min_quality_score
        self.budget_pacing_enabled = budget_pacing_enabled
        self.frequency_cap_enabled = frequency_cap_enabled

    def compute_ecpm(self, ad: Ad) -> float:
        """
        Compute effective Cost Per Mille (thousand impressions).

        eCPM = Expected revenue per 1000 impressions

        Formula:
            eCPM = pCTR * bid * quality_score * 1000

        Why quality_score?
        - High-quality ads get a boost in ranking
        - Low-quality ads get penalized
        - Balances revenue with user experience

        Args:
            ad: Ad candidate

        Returns:
            eCPM value ($)

        Example:
            >>> ad.predicted_ctr = 0.04  # 4% CTR
            >>> ad.bid_amount = 2.50     # $2.50 per click
            >>> ad.quality_score = 8.0   # High quality
            >>> ecpm = engine.compute_ecpm(ad)
            >>> # eCPM = 0.04 * 2.50 * 8.0 * 1000 = $800
        """
        return ad.predicted_ctr * ad.bid_amount * ad.quality_score * 1000.0

    def compute_ranking_score(self, ad: Ad) -> float:
        """
        Compute multi-objective ranking score.

        Balances three objectives:
        1. User engagement (CTR) - users click on relevant ads
        2. Conversion (CVR) - ads that drive actions
        3. Revenue (eCPM) - platform monetization

        Formula:
            Score = α * pCTR + β * pCVR + γ * (eCPM / 1000)

        This is a Pareto optimization problem. Weights (α, β, γ) are tuned
        via A/B testing to maximize long-term metrics.

        Args:
            ad: Ad candidate

        Returns:
            Ranking score (higher is better)

        Example:
            >>> ad.predicted_ctr = 0.04
            >>> ad.predicted_cvr = 0.01
            >>> ad.bid_amount = 2.00
            >>> ad.quality_score = 7.0
            >>> score = engine.compute_ranking_score(ad)
        """
        ecpm = self.compute_ecpm(ad)

        # Normalize eCPM to similar scale as CTR/CVR
        normalized_ecpm = ecpm / 1000.0

        # Multi-objective score
        score = (
            self.alpha * ad.predicted_ctr +
            self.beta * ad.predicted_cvr +
            self.gamma * normalized_ecpm
        )

        return score

    def apply_budget_pacing(self, ad: Ad, current_hour: int = 12) -> float:
        """
        Apply budget pacing to smooth ad spend over the day.

        Problem: Without pacing, ads exhaust budget in first few hours
        Solution: Throttle spend to match expected daily delivery

        Algorithm (Proportional Pacing):
            ideal_spend_rate = daily_budget / 24 hours
            actual_spend_rate = budget_spent_today / hours_elapsed
            pacing_multiplier = ideal_spend_rate / actual_spend_rate

        Args:
            ad: Ad candidate
            current_hour: Current hour of day (0-23)

        Returns:
            Pacing multiplier (0-1 scale)
                1.0 = on track (show ad normally)
                0.5 = overspending (throttle 50%)
                1.5 = underspending (boost 50%)

        Example:
            >>> ad.daily_budget = 1000.0
            >>> ad.budget_spent_today = 600.0
            >>> multiplier = engine.apply_budget_pacing(ad, current_hour=12)
            >>> # Should have spent: $500 (12/24 * $1000)
            >>> # Actually spent: $600
            >>> # Overspending, so throttle
        """
        if not self.budget_pacing_enabled:
            return 1.0

        # Hours elapsed in the day
        hours_elapsed = current_hour + 1  # 1-24 scale

        # Ideal spend rate (linear pacing)
        ideal_spend = (hours_elapsed / 24.0) * ad.daily_budget

        # Actual spend
        actual_spend = ad.budget_spent_today

        # Budget remaining
        budget_remaining = ad.daily_budget - actual_spend

        if budget_remaining <= 0:
            return 0.0  # Budget exhausted, don't show ad

        # Pacing multiplier
        if ideal_spend > 0:
            pacing_multiplier = min(1.5, ideal_spend / (actual_spend + 1e-6))
        else:
            pacing_multiplier = 1.0

        return pacing_multiplier

    def check_frequency_cap(
        self,
        ad: Ad,
        user_id: str,
        user_ad_impressions: Dict[str, int],
        max_impressions_per_day: int = 3
    ) -> bool:
        """
        Check if ad has exceeded frequency cap for this user.

        Frequency Cap: Maximum times a user sees the same ad per day
        - Prevents ad fatigue
        - Improves user experience
        - Can actually improve CTR (less repetition)

        Args:
            ad: Ad candidate
            user_id: User identifier
            user_ad_impressions: Dict of {ad_id: impression_count_today}
            max_impressions_per_day: Maximum impressions allowed

        Returns:
            True if ad can be shown, False if frequency cap exceeded

        Example:
            >>> impressions = {'ad_123': 2, 'ad_456': 5}
            >>> can_show = engine.check_frequency_cap(
            ...     ad, user_id='user_789', user_ad_impressions=impressions
            ... )
        """
        if not self.frequency_cap_enabled:
            return True

        ad_impressions = user_ad_impressions.get(ad.ad_id, 0)
        return ad_impressions < max_impressions_per_day

    def filter_eligible_ads(
        self,
        ads: List[Ad],
        user_id: str,
        user_ad_impressions: Dict[str, int],
        current_hour: int = 12
    ) -> List[Ad]:
        """
        Filter ads eligible to participate in auction.

        Filters:
        1. Quality score above minimum
        2. Budget not exhausted
        3. Frequency cap not exceeded

        Args:
            ads: All candidate ads
            user_id: User identifier
            user_ad_impressions: User's ad impression history
            current_hour: Current hour of day

        Returns:
            Eligible ads that can participate in auction

        Example:
            >>> all_ads = [ad1, ad2, ad3, ad4]
            >>> eligible = engine.filter_eligible_ads(
            ...     all_ads, user_id='123', user_ad_impressions={}
            ... )
            >>> # Maybe only [ad1, ad3] pass filters
        """
        eligible_ads = []

        for ad in ads:
            # Quality filter
            if ad.quality_score < self.min_quality_score:
                continue

            # Budget filter
            if ad.budget_spent_today >= ad.daily_budget:
                continue

            if ad.budget_spent_total >= ad.total_budget:
                continue

            # Frequency cap filter
            if not self.check_frequency_cap(ad, user_id, user_ad_impressions):
                continue

            eligible_ads.append(ad)

        return eligible_ads

    def run_auction(
        self,
        ads: List[Ad],
        user_id: str,
        user_ad_impressions: Optional[Dict[str, int]] = None,
        current_hour: int = 12
    ) -> Optional[AuctionResult]:
        """
        Run second-price (Vickrey) auction.

        Algorithm:
        1. Filter eligible ads (quality, budget, frequency)
        2. Compute ranking score for each ad
        3. Sort by ranking score (descending)
        4. Winner: highest scoring ad
        5. Price: what winner would need to bid to beat 2nd place + ε

        Second-Price Auction Properties:
        - Truthful bidding (dominant strategy to bid true value)
        - Stable revenue (less bid shading than first-price)
        - Efficient allocation (best ad wins)

        Args:
            ads: All candidate ads (after ML ranking)
            user_id: User identifier
            user_ad_impressions: User's ad impression history
            current_hour: Current hour of day (for pacing)

        Returns:
            AuctionResult with winner and pricing, or None if no eligible ads

        Example:
            >>> ads = [ad1, ad2, ad3]
            >>> result = engine.run_auction(ads, user_id='123')
            >>> if result:
            ...     print(f"Winner: {result.winning_ad.ad_id}")
            ...     print(f"Price: ${result.second_price:.4f}")
            ...     print(f"eCPM: ${result.ecpm:.2f}")
        """
        if user_ad_impressions is None:
            user_ad_impressions = {}

        # Filter eligible ads
        eligible_ads = self.filter_eligible_ads(
            ads, user_id, user_ad_impressions, current_hour
        )

        if len(eligible_ads) == 0:
            return None  # No eligible ads

        # Compute ranking scores
        ad_scores = []
        for ad in eligible_ads:
            score = self.compute_ranking_score(ad)

            # Apply budget pacing
            pacing_multiplier = self.apply_budget_pacing(ad, current_hour)
            adjusted_score = score * pacing_multiplier

            ad_scores.append((ad, adjusted_score))

        # Sort by score (descending)
        ad_scores.sort(key=lambda x: x[1], reverse=True)

        # Winner: highest scoring ad
        winning_ad, winning_score = ad_scores[0]

        # Second-price calculation
        if len(ad_scores) > 1:
            second_ad, second_score = ad_scores[1]

            # Price winner pays: minimum bid to beat 2nd place
            # Solve: pCTR_winner * price * quality_winner = score_second
            if winning_ad.predicted_ctr > 0 and winning_ad.quality_score > 0:
                second_price = (
                    second_score /
                    (winning_ad.predicted_ctr * winning_ad.quality_score * self.gamma)
                )
            else:
                second_price = winning_ad.bid_amount

            # Add small increment (1 cent)
            second_price = min(second_price + 0.01, winning_ad.bid_amount)
        else:
            # Only one ad, pay minimum (reserve price)
            second_price = 0.01  # 1 cent reserve price

        # Compute final eCPM
        ecpm = winning_ad.predicted_ctr * second_price * winning_ad.quality_score * 1000.0

        return AuctionResult(
            winning_ad=winning_ad,
            second_price=second_price,
            ecpm=ecpm,
            rank_score=winning_score,
            all_candidates=ad_scores
        )


class BudgetOptimizer:
    """
    Campaign budget optimization for advertisers.

    Helps advertisers allocate budget across multiple campaigns to maximize ROI.

    Features:
    - Bid shading detection and correction
    - Budget allocation across campaigns
    - Pacing recommendations
    - Performance forecasting

    Example:
        >>> optimizer = BudgetOptimizer()
        >>> campaigns = [campaign1, campaign2, campaign3]
        >>> recommendations = optimizer.optimize_budget(campaigns, total_budget=10000)
    """

    def __init__(self):
        pass

    def compute_expected_roi(
        self,
        predicted_ctr: float,
        predicted_cvr: float,
        bid_amount: float,
        avg_order_value: float
    ) -> float:
        """
        Compute expected Return on Investment.

        ROI = (Revenue - Cost) / Cost

        Args:
            predicted_ctr: CTR prediction
            predicted_cvr: Conversion rate prediction
            bid_amount: Cost per click
            avg_order_value: Average revenue per conversion

        Returns:
            Expected ROI (e.g., 2.0 = 200% ROI)

        Example:
            >>> roi = optimizer.compute_expected_roi(
            ...     predicted_ctr=0.04,
            ...     predicted_cvr=0.02,
            ...     bid_amount=2.00,
            ...     avg_order_value=100.00
            ... )
        """
        # Expected clicks per 1000 impressions
        expected_clicks = predicted_ctr * 1000

        # Expected conversions per 1000 impressions
        expected_conversions = predicted_ctr * predicted_cvr * 1000

        # Cost per 1000 impressions
        cost = expected_clicks * bid_amount

        # Revenue per 1000 impressions
        revenue = expected_conversions * avg_order_value

        if cost > 0:
            roi = (revenue - cost) / cost
        else:
            roi = 0.0

        return roi


# Example usage and testing
if __name__ == "__main__":
    print("=== Ad Auction Engine Example ===\n")

    # Create auction engine
    engine = AdAuctionEngine(
        alpha=0.4,  # 40% weight on user engagement (CTR)
        beta=0.3,   # 30% weight on conversion (CVR)
        gamma=0.3,  # 30% weight on revenue (eCPM)
        min_quality_score=3.0
    )

    # Create example ads
    ads = [
        Ad(
            ad_id="ad_001",
            advertiser_id="advertiser_A",
            campaign_id="campaign_1",
            creative_text="Gaming Headset - 50% Off!",
            bid_amount=2.50,
            bidding_strategy=BiddingStrategy.CPC,
            daily_budget=1000.0,
            total_budget=10000.0,
            budget_spent_today=200.0,
            budget_spent_total=2000.0,
            quality_score=8.5,
            predicted_ctr=0.045,
            predicted_cvr=0.012,
            category="gaming"
        ),
        Ad(
            ad_id="ad_002",
            advertiser_id="advertiser_B",
            campaign_id="campaign_2",
            creative_text="New Action RPG Game Release",
            bid_amount=3.00,
            bidding_strategy=BiddingStrategy.CPC,
            daily_budget=1500.0,
            total_budget=15000.0,
            budget_spent_today=800.0,
            budget_spent_total=5000.0,
            quality_score=7.0,
            predicted_ctr=0.038,
            predicted_cvr=0.015,
            category="gaming"
        ),
        Ad(
            ad_id="ad_003",
            advertiser_id="advertiser_C",
            campaign_id="campaign_3",
            creative_text="Premium Audio Equipment",
            bid_amount=1.80,
            bidding_strategy=BiddingStrategy.CPC,
            daily_budget=800.0,
            total_budget=8000.0,
            budget_spent_today=100.0,
            budget_spent_total=1000.0,
            quality_score=9.0,
            predicted_ctr=0.042,
            predicted_cvr=0.010,
            category="electronics"
        )
    ]

    # Run auction
    print("Running auction for user_123...\n")
    result = engine.run_auction(
        ads=ads,
        user_id="user_123",
        user_ad_impressions={},
        current_hour=12
    )

    if result:
        print(f"🏆 Winner: {result.winning_ad.ad_id}")
        print(f"   Creative: {result.winning_ad.creative_text}")
        print(f"   Original Bid: ${result.winning_ad.bid_amount:.2f}")
        print(f"   Second Price: ${result.second_price:.2f}")
        print(f"   eCPM: ${result.ecpm:.2f}")
        print(f"   Ranking Score: {result.rank_score:.4f}")
        print(f"   Quality Score: {result.winning_ad.quality_score}")
        print(f"\n📊 All Candidates:")

        for i, (ad, score) in enumerate(result.all_candidates, 1):
            ecpm = engine.compute_ecpm(ad)
            print(f"   {i}. {ad.ad_id}: Score={score:.4f}, eCPM=${ecpm:.2f}, CTR={ad.predicted_ctr:.3f}")

    print("\n=== Budget Pacing Example ===\n")
    for ad in ads:
        multiplier = engine.apply_budget_pacing(ad, current_hour=12)
        print(f"{ad.ad_id}:")
        print(f"  Daily Budget: ${ad.daily_budget:.2f}")
        print(f"  Spent Today: ${ad.budget_spent_today:.2f}")
        print(f"  Pacing Multiplier: {multiplier:.2f}x")
        if multiplier < 1.0:
            print(f"  → Overspending, throttle ads")
        elif multiplier > 1.0:
            print(f"  → Underspending, boost ads")
        print()
