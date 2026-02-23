"""
AI-Powered Fraud Investigation Service
Generates investigation reports for flagged transactions using Claude API

This extends detection (what you built) to investigation (what Finic does)
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import requests


class InvestigationAgent:
    """
    AI agent that investigates flagged transactions and generates case summaries.
    
    Inspired by Finic's approach:
    - Analyzes transaction context
    - Pulls user history
    - Summarizes risk factors
    - Recommends action (block/allow/escalate)
    """
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model = "claude-sonnet-4-20250514"
        
    def investigate(
        self, 
        transaction: Dict,
        user_history: List[Dict],
        risk_factors: List[Dict],
        peer_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Generate an investigation report for a flagged transaction.
        
        Args:
            transaction: The flagged transaction
            user_history: Past transactions from this user
            risk_factors: Detected risk signals
            peer_stats: Department/peer spending statistics
            
        Returns:
            Investigation report with recommendation
        """
        
        if not self.api_key:
            return self._generate_fallback_report(transaction, risk_factors)
        
        # Build investigation context
        context = self._build_context(transaction, user_history, risk_factors, peer_stats)
        
        # Generate report using Claude
        report = self._call_claude(context)
        
        return report
    
    def _build_context(
        self, 
        transaction: Dict,
        user_history: List[Dict],
        risk_factors: List[Dict],
        peer_stats: Optional[Dict]
    ) -> str:
        """Build context prompt for Claude."""
        
        # Transaction summary
        txn_summary = f"""
TRANSACTION UNDER INVESTIGATION:
- ID: {transaction.get('id', 'N/A')}
- Amount: ${transaction.get('amount', 0):,.2f}
- Merchant: {transaction.get('merchant_name', 'Unknown')}
- Category (MCC): {transaction.get('merchant_category_code', 'N/A')}
- User ID: {transaction.get('user_id', 'N/A')}
- Department: {transaction.get('department', 'N/A')}
- Date/Time: {transaction.get('transaction_date', 'N/A')}
- Risk Score: {transaction.get('risk_score', 0):.1f}/100
"""
        
        # Risk factors
        risk_summary = "\nDETECTED RISK FACTORS:\n"
        for factor in risk_factors:
            risk_summary += f"- {factor.get('signal', 'Unknown')}: {factor.get('reason', 'No details')} (Score: {factor.get('score', 0)})\n"
        
        # User history summary
        if user_history:
            avg_amount = sum(t.get('amount', 0) for t in user_history) / len(user_history)
            max_amount = max(t.get('amount', 0) for t in user_history)
            total_txns = len(user_history)
            
            # Find common merchants
            merchants = [t.get('merchant_name') for t in user_history if t.get('merchant_name')]
            merchant_counts = {}
            for m in merchants:
                merchant_counts[m] = merchant_counts.get(m, 0) + 1
            top_merchants = sorted(merchant_counts.items(), key=lambda x: -x[1])[:5]
            
            history_summary = f"""
USER TRANSACTION HISTORY:
- Total Past Transactions: {total_txns}
- Average Transaction: ${avg_amount:,.2f}
- Largest Transaction: ${max_amount:,.2f}
- Top Merchants: {', '.join([m[0] for m in top_merchants]) if top_merchants else 'N/A'}
- This transaction is {transaction.get('amount', 0) / avg_amount:.1f}x their average
"""
        else:
            history_summary = "\nUSER TRANSACTION HISTORY:\n- No prior transaction history available (new user)\n"
        
        # Peer comparison
        if peer_stats:
            peer_summary = f"""
DEPARTMENT PEER COMPARISON:
- Department: {peer_stats.get('department', 'N/A')}
- Peer Average: ${peer_stats.get('avg_amount', 0):,.2f}
- Peer Max: ${peer_stats.get('max_amount', 0):,.2f}
- This transaction vs peer average: {transaction.get('amount', 0) / max(peer_stats.get('avg_amount', 1), 1):.1f}x
"""
        else:
            peer_summary = ""
        
        return txn_summary + risk_summary + history_summary + peer_summary
    
    def _call_claude(self, context: str) -> Dict:
        """Call Claude API to generate investigation report."""
        
        system_prompt = """You are a fraud investigation analyst at a fintech company. Your job is to review flagged transactions and generate clear, actionable investigation reports.

For each case, you must:
1. Summarize the key facts
2. Analyze the risk factors in context
3. Consider the user's history and patterns
4. Make a clear recommendation: APPROVE, BLOCK, or ESCALATE

Be concise but thorough. Focus on actionable insights."""

        user_prompt = f"""Review this flagged transaction and generate an investigation report:

{context}

Provide your analysis in this format:
1. EXECUTIVE SUMMARY (2-3 sentences)
2. KEY FINDINGS (bullet points)
3. RISK ASSESSMENT (Low/Medium/High/Critical with justification)
4. RECOMMENDATION (APPROVE/BLOCK/ESCALATE with reasoning)
5. SUGGESTED FOLLOW-UP ACTIONS (if any)"""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ],
                    "system": system_prompt
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                report_text = result['content'][0]['text']
                
                return {
                    "status": "complete",
                    "generated_at": datetime.now().isoformat(),
                    "model": self.model,
                    "report": report_text,
                    "recommendation": self._extract_recommendation(report_text),
                    "risk_level": self._extract_risk_level(report_text)
                }
            else:
                return self._generate_fallback_report(
                    {"error": f"API returned {response.status_code}"},
                    []
                )
                
        except Exception as e:
            return self._generate_fallback_report(
                {"error": str(e)},
                []
            )
    
    def _extract_recommendation(self, report: str) -> str:
        """Extract recommendation from report text."""
        report_upper = report.upper()
        if "RECOMMENDATION: BLOCK" in report_upper or "RECOMMEND BLOCKING" in report_upper:
            return "BLOCK"
        elif "RECOMMENDATION: ESCALATE" in report_upper or "RECOMMEND ESCALAT" in report_upper:
            return "ESCALATE"
        elif "RECOMMENDATION: APPROVE" in report_upper or "RECOMMEND APPROV" in report_upper:
            return "APPROVE"
        return "REVIEW"
    
    def _extract_risk_level(self, report: str) -> str:
        """Extract risk level from report text."""
        report_upper = report.upper()
        if "CRITICAL" in report_upper:
            return "critical"
        elif "HIGH" in report_upper and "RISK" in report_upper:
            return "high"
        elif "MEDIUM" in report_upper and "RISK" in report_upper:
            return "medium"
        return "low"
    
    def _generate_fallback_report(self, transaction: Dict, risk_factors: List[Dict]) -> Dict:
        """Generate a rule-based report when Claude API is unavailable."""
        
        amount = transaction.get('amount', 0)
        risk_score = transaction.get('risk_score', 0)
        
        # Determine recommendation based on rules
        if risk_score >= 80:
            recommendation = "BLOCK"
            risk_level = "critical"
        elif risk_score >= 60:
            recommendation = "ESCALATE"
            risk_level = "high"
        elif risk_score >= 40:
            recommendation = "REVIEW"
            risk_level = "medium"
        else:
            recommendation = "APPROVE"
            risk_level = "low"
        
        # Build findings
        findings = []
        for factor in risk_factors:
            findings.append(f"- {factor.get('signal', 'Unknown')}: {factor.get('reason', 'Detected')}")
        
        report = f"""## EXECUTIVE SUMMARY
Transaction of ${amount:,.2f} flagged with risk score {risk_score:.1f}/100. {'Multiple risk factors detected requiring review.' if len(risk_factors) > 2 else 'Limited risk factors detected.'}

## KEY FINDINGS
{chr(10).join(findings) if findings else '- No specific risk factors documented'}

## RISK ASSESSMENT
**{risk_level.upper()}** - Based on composite risk score of {risk_score:.1f}/100

## RECOMMENDATION
**{recommendation}** - {'Transaction exceeds risk thresholds' if recommendation == 'BLOCK' else 'Manual review recommended' if recommendation in ['ESCALATE', 'REVIEW'] else 'Transaction within acceptable parameters'}

## SUGGESTED FOLLOW-UP
{'- Contact cardholder to verify transaction' if recommendation != 'APPROVE' else '- No action required'}
{'- Review merchant for potential blacklisting' if risk_score >= 70 else ''}
"""
        
        return {
            "status": "complete",
            "generated_at": datetime.now().isoformat(),
            "model": "rule-based-fallback",
            "report": report,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "note": "Generated using rule-based fallback (Claude API key not configured)"
        }


# Singleton instance
investigation_agent = InvestigationAgent()
