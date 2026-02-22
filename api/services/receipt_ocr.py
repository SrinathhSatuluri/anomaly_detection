"""
Receipt OCR - Extract data from receipt images
Simplified version of Ramp's receipt intelligence
"""

import re
import os
from typing import Dict, Optional, Tuple, List
from datetime import datetime

try:
    from PIL import Image
    import pytesseract

    # Windows Tesseract path
    if os.name == 'nt':
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class ReceiptOCR:
    """Extract structured data from receipt images."""

    def __init__(self):
        # Common patterns
        self.amount_pattern = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        self.date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
        ]
        self.total_keywords = ['total', 'amount due', 'balance due', 'grand total', 'sum']

    def is_available(self) -> bool:
        """Check if OCR is available."""
        return OCR_AVAILABLE

    def extract_from_image(self, image_path: str) -> Dict:
        """Extract receipt data from image."""
        if not OCR_AVAILABLE:
            return {'error': 'OCR not available. Install pytesseract and Tesseract-OCR.'}

        try:
            if not os.path.exists(image_path):
                return {'error': f'Image not found: {image_path}'}

            image = Image.open(image_path)

            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')

            # Extract text
            text = pytesseract.image_to_string(image)

            return {
                'success': True,
                'raw_text': text,
                'line_count': len(text.split('\n')),
                'amounts': self._extract_amounts(text),
                'total': self._extract_total(text),
                'subtotal': self._extract_subtotal(text),
                'tax': self._extract_tax(text),
                'date': self._extract_date(text),
                'merchant': self._extract_merchant(text),
                'items': self._extract_line_items(text),
                'confidence': self._calculate_confidence(text)
            }
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _extract_amounts(self, text: str) -> List[float]:
        """Find all dollar amounts in text."""
        amounts = re.findall(self.amount_pattern, text)
        parsed = []
        for a in amounts:
            try:
                val = float(a.replace('$', '').replace(',', ''))
                if val > 0:
                    parsed.append(val)
            except:
                pass
        return sorted(set(parsed))

    def _extract_total(self, text: str) -> Optional[float]:
        """Find the total amount."""
        lines = text.upper().split('\n')

        # Look for lines with "TOTAL" keyword
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in self.total_keywords):
                if 'sub' not in line_lower:  # Skip subtotal
                    amounts = re.findall(self.amount_pattern, line)
                    if amounts:
                        try:
                            return float(amounts[-1].replace('$', '').replace(',', ''))
                        except:
                            pass

        # Fall back to largest amount
        all_amounts = self._extract_amounts(text)
        return max(all_amounts) if all_amounts else None

    def _extract_subtotal(self, text: str) -> Optional[float]:
        """Find subtotal amount."""
        lines = text.lower().split('\n')

        for line in lines:
            if 'subtotal' in line or 'sub total' in line:
                amounts = re.findall(self.amount_pattern, line)
                if amounts:
                    try:
                        return float(amounts[-1].replace('$', '').replace(',', ''))
                    except:
                        pass
        return None

    def _extract_tax(self, text: str) -> Optional[float]:
        """Find tax amount."""
        lines = text.lower().split('\n')

        for line in lines:
            if 'tax' in line and 'total' not in line:
                amounts = re.findall(self.amount_pattern, line)
                if amounts:
                    try:
                        return float(amounts[-1].replace('$', '').replace(',', ''))
                    except:
                        pass
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """Find date on receipt."""
        for pattern in self.date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            if dates:
                return dates[0]
        return None

    def _extract_merchant(self, text: str) -> Optional[str]:
        """Extract merchant name (usually first non-empty lines)."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # Skip very short lines (likely noise)
        for line in lines[:5]:
            if len(line) >= 3 and not re.match(r'^[\d\s\-\.\$]+$', line):
                return line

        return lines[0] if lines else None

    def _extract_line_items(self, text: str) -> List[Dict]:
        """Extract individual line items with prices."""
        items = []
        lines = text.split('\n')

        for line in lines:
            # Look for lines with text followed by amount
            amounts = re.findall(self.amount_pattern, line)
            if amounts and len(line) > 10:
                # Remove the amount to get item description
                item_text = re.sub(self.amount_pattern, '', line).strip()
                item_text = re.sub(r'[^\w\s]', '', item_text).strip()

                if item_text and len(item_text) >= 3:
                    try:
                        price = float(amounts[-1].replace('$', '').replace(',', ''))
                        if 0 < price < 10000:  # Reasonable item price
                            items.append({
                                'description': item_text[:50],
                                'price': price
                            })
                    except:
                        pass

        return items[:20]  # Limit to 20 items

    def _calculate_confidence(self, text: str) -> float:
        """Estimate extraction confidence."""
        factors = []

        # Has total
        has_total = any(kw in text.lower() for kw in self.total_keywords)
        factors.append(1.0 if has_total else 0.0)

        # Has amounts
        has_amounts = bool(re.search(self.amount_pattern, text))
        factors.append(1.0 if has_amounts else 0.0)

        # Has date
        has_date = any(re.search(p, text, re.IGNORECASE) for p in self.date_patterns)
        factors.append(1.0 if has_date else 0.0)

        # Reasonable text length
        reasonable_length = 50 < len(text) < 5000
        factors.append(1.0 if reasonable_length else 0.5)

        # Has multiple lines
        line_count = len(text.split('\n'))
        factors.append(1.0 if line_count > 5 else 0.5)

        return round(sum(factors) / len(factors), 2)

    def verify_transaction(self, image_path: str,
                          claimed_amount: float,
                          claimed_merchant: str = None,
                          tolerance: float = 1.00) -> Tuple[bool, Dict]:
        """
        Verify receipt matches transaction claim.

        Args:
            image_path: Path to receipt image
            claimed_amount: Amount claimed in expense report
            claimed_merchant: Merchant name claimed (optional)
            tolerance: Dollar tolerance for amount matching

        Returns:
            (is_verified: bool, details: dict)
        """
        extracted = self.extract_from_image(image_path)

        if not extracted.get('success'):
            return False, {
                'verified': False,
                'error': extracted.get('error', 'Extraction failed'),
                'issues': ['Could not process receipt image']
            }

        issues = []
        warnings = []

        # Check amount
        receipt_total = extracted.get('total')
        if receipt_total:
            diff = abs(receipt_total - claimed_amount)
            if diff > tolerance:
                issues.append({
                    'type': 'amount_mismatch',
                    'severity': 'high' if diff > 10 else 'medium',
                    'message': f"Receipt shows ${receipt_total:.2f}, claimed ${claimed_amount:.2f} (diff: ${diff:.2f})"
                })
        else:
            warnings.append("Could not extract total from receipt")

        # Check merchant if provided
        receipt_merchant = extracted.get('merchant')
        if claimed_merchant and receipt_merchant:
            claimed_lower = claimed_merchant.lower()
            receipt_lower = receipt_merchant.lower()

            # Fuzzy match - check if any significant words match
            claimed_words = set(claimed_lower.split())
            receipt_words = set(receipt_lower.split())

            if not (claimed_words & receipt_words) and claimed_lower not in receipt_lower:
                issues.append({
                    'type': 'merchant_mismatch',
                    'severity': 'medium',
                    'message': f"Receipt merchant '{receipt_merchant}' doesn't match claimed '{claimed_merchant}'"
                })

        # Check for suspicious patterns
        if extracted.get('confidence', 0) < 0.5:
            warnings.append("Low confidence in receipt extraction - image may be unclear")

        is_verified = len(issues) == 0

        return is_verified, {
            'verified': is_verified,
            'issues': issues,
            'warnings': warnings,
            'extracted': {
                'total': receipt_total,
                'merchant': receipt_merchant,
                'date': extracted.get('date'),
                'item_count': len(extracted.get('items', [])),
                'confidence': extracted.get('confidence')
            },
            'claimed': {
                'amount': claimed_amount,
                'merchant': claimed_merchant
            }
        }


# Singleton instance
receipt_ocr = ReceiptOCR()