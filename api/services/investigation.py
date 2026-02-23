"""
AI-Powered Fraud Investigation Service
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import requests


class InvestigationAgent:
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.model = "claude-sonnet-4-20250514"

    def investigate(self, transaction: Dict, user_history: List[Dict], risk_factors: List[Dict], peer_stats: Optional[Dict] = None) -> Dict:
        if not self.api_key:
            return self._generate_fallback_report(transaction, risk_factors)
        context = self._build_context(transaction, user_history, risk_factors, peer_stats)
        return self._call_claude(context)

    def _build_context(self, transaction: Dict, user_history: List[Dict], risk_factors: List[Dict], peer_stats: Optional[Dict]) -> str:
        txn_summary = f"""
TRANSACTION UNDER INVESTIGATION:
- ID: {transaction.get('id', 'N/A')}
- Amount: ${transaction.get('amount', 0):,.2f}
- Merchant: {transaction.get('merchant_name', 'Unknown')}
- Category (MCC): {transaction.get('merchant_category_code', 'N/A')}
- User ID: {transaction.get('user_id', 'N/A')}
- Department: {transaction.get('department', 'N/A')}
- Risk Score: {transaction.get('risk_score', 0):.1f}/100
"""
        risk_summary = "\nDETECTED RISK FACTORS:\n"
        for factor in risk_factors:
            risk_summary += f"- {factor.get('signal', 'Unknown')}: {factor.get('reason', 'No details')} (Score: {factor.get('score', 0)})\n"

        if user_history:
            avg_amount = sum(t.get('amount', 0) for t in user_history) / len(user_history)
            history_summary = f"\nUSER HISTORY: {len(user_history)} transactions, avg ${avg_amount:,.2f}\n"
        else:
            history_summary = "\nUSER HISTORY: No prior history (new user)\n"

        return txn_summary + risk_summary + history_summary

    def _call_claude(self, context: str) -> Dict:
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": self.model, "max_tokens": 1024, "messages": [{"role": "user", "content": f"Review this flagged transaction and provide: 1) EXECUTIVE SUMMARY 2) KEY FINDINGS 3) RISK ASSESSMENT 4) RECOMMENDATION (APPROVE/BLOCK/ESCALATE)\n\n{context}"}]},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                report_text = result['content'][0]['text']
                return {"status": "complete", "generated_at": datetime.now().isoformat(), "model": self.model, "report": report_text, "recommendation": self._extract_recommendation(report_text)}
        except Exception as e:
            pass
        return self._generate_fallback_report({}, [])

    def _extract_recommendation(self, report: str) -> str:
        report_upper = report.upper()
        if "BLOCK" in report_upper: return "BLOCK"
        elif "ESCALATE" in report_upper: return "ESCALATE"
        return "APPROVE"

    def _generate_fallback_report(self, transaction: Dict, risk_factors: List[Dict]) -> Dict:
        amount = transaction.get('amount', 0)
        risk_score = transaction.get('risk_score', 0)

        if risk_score >= 80: recommendation, risk_level = "BLOCK", "critical"
        elif risk_score >= 60: recommendation, risk_level = "ESCALATE", "high"
        elif risk_score >= 40: recommendation, risk_level = "REVIEW", "medium"
        else: recommendation, risk_level = "APPROVE", "low"

        findings = [f"- {f.get('signal')}: {f.get('reason')}" for f in risk_factors]

        report = f"""## EXECUTIVE SUMMARY
Transaction of ${amount:,.2f} flagged with risk score {risk_score:.1f}/100.

## KEY FINDINGS
{chr(10).join(findings) if findings else '- No specific risk factors'}

## RISK ASSESSMENT
**{risk_level.upper()}** - Risk score: {risk_score:.1f}/100

## RECOMMENDATION
**{recommendation}**
"""
        return {"status": "complete", "generated_at": datetime.now().isoformat(), "model": "rule-based-fallback", "report": report, "recommendation": recommendation, "risk_level": risk_level}


investigation_agent = InvestigationAgent()