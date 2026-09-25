"""Text normalization and preprocessing for business names and addresses."""

import re
import unicodedata
from typing import List, Set, Dict, Optional
from functools import lru_cache

# Common business suffixes and their abbreviations
BUSINESS_SUFFIXES = {
    'incorporated': 'inc',
    'inc': 'inc',
    'corporation': 'corp',
    'corp': 'corp',
    'company': 'co',
    'co': 'co',
    'limited': 'ltd',
    'ltd': 'ltd',
    'limited liability company': 'llc',
    'llc': 'llc',
    'limited partnership': 'lp',
    'lp': 'lp',
    'limited liability partnership': 'llp',
    'llp': 'llp',
    'professional corporation': 'pc',
    'pc': 'pc',
    'professional association': 'pa',
    'pa': 'pa',
    'enterprises': 'ent',
    'enterprise': 'ent',
    'ent': 'ent',
    'group': 'grp',
    'grp': 'grp',
    'holdings': 'hldgs',
    'holding': 'hldgs',
    'hldgs': 'hldgs',
    'international': 'intl',
    'intl': 'intl',
    'national': 'natl',
    'natl': 'natl',
    'associates': 'assoc',
    'associate': 'assoc',
    'assoc': 'assoc',
    'services': 'svcs',
    'service': 'svcs',
    'svcs': 'svcs',
    'solutions': 'sol',
    'solution': 'sol',
    'sol': 'sol',
    'technologies': 'tech',
    'technology': 'tech',
    'tech': 'tech',
    'systems': 'sys',
    'system': 'sys',
    'sys': 'sys',
    'industries': 'ind',
    'industry': 'ind',
    'ind': 'ind',
    'manufacturing': 'mfg',
    'mfg': 'mfg',
    'distributors': 'dist',
    'distributor': 'dist',
    'dist': 'dist',
    'wholesale': 'whol',
    'whol': 'whol',
    'retail': 'ret',
    'ret': 'ret',
}

# Address abbreviations
ADDRESS_ABBREVIATIONS = {
    'street': 'st',
    'st': 'st',
    'avenue': 'ave',
    'ave': 'ave',
    'av': 'ave',
    'road': 'rd',
    'rd': 'rd',
    'drive': 'dr',
    'dr': 'dr',
    'lane': 'ln',
    'ln': 'ln',
    'court': 'ct',
    'ct': 'ct',
    'boulevard': 'blvd',
    'blvd': 'blvd',
    'blv': 'blvd',
    'place': 'pl',
    'pl': 'pl',
    'circle': 'cir',
    'cir': 'cir',
    'way': 'way',
    'wy': 'way',
    'terrace': 'ter',
    'ter': 'ter',
    'highway': 'hwy',
    'hwy': 'hwy',
    'parkway': 'pkwy',
    'pkwy': 'pkwy',
    'expressway': 'expy',
    'expy': 'expy',
    'square': 'sq',
    'sq': 'sq',
    'plaza': 'plz',
    'plz': 'plz',
    'center': 'ctr',
    'ctr': 'ctr',
    'centre': 'ctr',
    'building': 'bldg',
    'bldg': 'bldg',
    'suite': 'ste',
    'ste': 'ste',
    'unit': 'unit',
    'apartment': 'apt',
    'apt': 'apt',
    'floor': 'fl',
    'fl': 'fl',
    'room': 'rm',
    'rm': 'rm',
    'north': 'n',
    'n': 'n',
    'south': 's',
    's': 's',
    'east': 'e',
    'e': 'e',
    'west': 'w',
    'w': 'w',
    'northeast': 'ne',
    'ne': 'ne',
    'northwest': 'nw',
    'nw': 'nw',
    'southeast': 'se',
    'se': 'se',
    'southwest': 'sw',
    'sw': 'sw',
    'number': '#',
    'no': '#',
    'near': 'nr',
    'nr': 'nr',
    'opposite': 'opp',
    'opp': 'opp',
    'behind': 'bhd',
    'bhd': 'bhd',
    'beside': 'bsd',
    'bsd': 'bsd',
    'landmark': 'lm',
    'lm': 'lm',
}

# Common stopwords for business names
NAME_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'by',
    'with', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'shall', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine',
    'yours', 'hers', 'ours', 'theirs'
}

# Address stopwords
ADDRESS_STOPWORDS = {
    'near', 'nr', 'opp', 'opposite', 'behind', 'bhd', 'beside', 'bsd',
    'landmark', 'lm', 'area', 'zone', 'sector', 'block', 'phase',
    'extension', 'ext', 'colony', 'col', 'nagar', 'vihar', 'puram',
    'enclave', 'estate', 'township', 'complex', 'tower', 'building',
    'bldg', 'floor', 'fl', 'wing', 'side', 'corner', 'junction',
    'cross', 'main', 'road', 'rd', 'street', 'st', 'avenue', 'ave'
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (NFKC)."""
    return unicodedata.normalize('NFKC', text)


def remove_accents(text: str) -> str:
    """Remove diacritical marks."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])


def lowercase(text: str) -> str:
    """Convert to lowercase."""
    return text.lower()


def remove_punctuation(text: str, keep: str = '') -> str:
    """Remove punctuation except specified characters."""
    punct = r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for ch in keep:
        punct = punct.replace(ch, '')
    return text.translate(str.maketrans('', '', punct))


def replace_punctuation_with_space(text: str) -> str:
    """Replace punctuation with spaces to preserve token boundaries."""
    punct = r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return text.translate(str.maketrans(punct, ' ' * len(punct)))


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single space."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_extra_spaces(text: str) -> str:
    """Remove leading/trailing and duplicate spaces."""
    return ' '.join(text.split())


def expand_abbreviations(text: str, abbr_dict: Dict[str, str]) -> str:
    """Expand abbreviations using provided dictionary."""
    words = text.split()
    expanded = [abbr_dict.get(w, w) for w in words]
    return ' '.join(expanded)


def standardize_business_suffixes(text: str) -> str:
    """Standardize business suffixes to canonical form."""
    words = text.split()
    if not words:
        return text
    
    # Check last few words for business suffixes
    for i in range(len(words) - 1, max(-1, len(words) - 4), -1):
        word = words[i].lower().rstrip('.,')
        if word in BUSINESS_SUFFIXES:
            words[i] = BUSINESS_SUFFIXES[word]
    
    return ' '.join(words)


def standardize_address_abbreviations(text: str) -> str:
    """Standardize address abbreviations to canonical form."""
    words = text.split()
    for i, word in enumerate(words):
        word_lower = word.lower().rstrip('.,')
        if word_lower in ADDRESS_ABBREVIATIONS:
            words[i] = ADDRESS_ABBREVIATIONS[word_lower]
    return ' '.join(words)


def extract_tokens(text: str, min_length: int = 2, stopwords: Optional[Set[str]] = None) -> List[str]:
    """Extract meaningful tokens from text."""
    text = replace_punctuation_with_space(text)
    tokens = text.lower().split()
    
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    
    tokens = [t for t in tokens if len(t) >= min_length]
    return tokens


def extract_ngrams(text: str, n: int = 3, min_freq: int = 1) -> List[str]:
    """Extract character n-grams from text."""
    text = re.sub(r'\s+', '', text.lower())
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def extract_postal_code(text: str) -> Optional[str]:
    """Extract postal code from address."""
    # US ZIP codes (5 or 9 digits)
    us_zip = re.search(r'\b\d{5}(?:-\d{4})?\b', text)
    if us_zip:
        return us_zip.group()
    
    # Canadian postal codes (A1A 1A1)
    ca_postal = re.search(r'\b[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d\b', text)
    if ca_postal:
        return ca_postal.group().replace(' ', '').upper()
    
    # UK postcodes
    uk_postal = re.search(r'\b[A-Za-z]{1,2}\d[A-Za-z\d]?\s?\d[A-Za-z]{2}\b', text)
    if uk_postal:
        return uk_postal.group().replace(' ', '').upper()
    
    # Generic: 4-6 digit codes
    generic = re.search(r'\b\d{4,6}\b', text)
    if generic:
        return generic.group()
    
    return None


def extract_house_number(text: str) -> Optional[str]:
    """Extract house/building number from address."""
    # Match patterns like "123", "123A", "123-45"
    match = re.search(r'\b(\d+[A-Za-z]?(?:-\d+)?)\b', text)
    if match:
        return match.group(1)
    return None


def extract_street_name(text: str) -> str:
    """Extract street name from address (remove house number, city, etc.)."""
    # Remove house number at start
    text = re.sub(r'^\d+[A-Za-z]?(?:-\d+)?\s+', '', text)
    # Remove suite/unit info
    text = re.sub(r'\s+(?:ste|suite|unit|apt|apartment|#)\s+[\w-]+', '', text, flags=re.IGNORECASE)
    return text.strip()


def parse_address_components(text: str) -> Dict[str, str]:
    """Parse address into components: house_number, street, city, state, postal_code."""
    components = {
        'house_number': '',
        'street': '',
        'city': '',
        'state': '',
        'postal_code': '',
        'raw': text
    }
    
    # Extract postal code
    postal = extract_postal_code(text)
    if postal:
        components['postal_code'] = postal
        text = text.replace(postal, '')
    
    # Extract house number
    house_num = extract_house_number(text)
    if house_num:
        components['house_number'] = house_num
    
    # Extract street (remaining after house number)
    components['street'] = extract_street_name(text)
    
    # Try to extract city/state (last parts before postal code)
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) >= 2:
        components['city'] = parts[-2]
        components['state'] = parts[-1]
    elif len(parts) == 1:
        components['city'] = parts[0]
    
    return components


def normalize_name(text: str) -> str:
    """Full normalization pipeline for business names."""
    if not text:
        return ''
    
    text = normalize_unicode(text)
    text = remove_accents(text)
    text = lowercase(text)
    text = replace_punctuation_with_space(text)
    text = standardize_business_suffixes(text)
    text = collapse_whitespace(text)
    return text


def normalize_address(text: str) -> str:
    """Full normalization pipeline for addresses."""
    if not text:
        return ''
    
    text = normalize_unicode(text)
    text = remove_accents(text)
    text = lowercase(text)
    text = replace_punctuation_with_space(text)
    text = standardize_address_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def get_name_blocking_keys(name: str, config: dict) -> List[str]:
    """Generate multiple blocking keys for a business name."""
    keys = []
    normalized = normalize_name(name)
    tokens = extract_tokens(normalized, min_length=config.get('name_token_min_length', 3),
                           stopwords=NAME_STOPWORDS)
    
    if not tokens:
        return ['_empty_']
    
    # First token
    if config.get('use_first_token', True):
        keys.append(f"name_first:{tokens[0]}")
    
    # First two tokens combined
    if len(tokens) >= 2 and config.get('use_first_two_tokens', True):
        keys.append(f"name_first2:{tokens[0]}_{tokens[1]}")
    
    # Individual tokens (for inverted index)
    if config.get('use_individual_tokens', True):
        for token in tokens[:config.get('name_token_max_tokens', 5)]:
            keys.append(f"name_token:{token}")
    
    # N-grams
    ngram_size = config.get('name_ngram_size', 3)
    if config.get('use_ngrams', True):
        for token in tokens[:3]:  # Top 3 tokens
            ngrams = extract_ngrams(token, n=ngram_size)
            for ng in ngrams[:5]:  # Limit ngrams per token
                keys.append(f"name_ngram:{ng}")
    
    # Phonetic keys
    if config.get('use_soundex', True):
        keys.append(f"name_soundex:{soundex(normalized)}")
    if config.get('use_metaphone', True):
        keys.append(f"name_metaphone:{metaphone(normalized)}")
    if config.get('use_nysiis', True):
        keys.append(f"name_nysiis:{nysiis(normalized)}")
    
    return keys


def get_address_blocking_keys(address: str, config: dict) -> List[str]:
    """Generate multiple blocking keys for an address."""
    keys = []
    normalized = normalize_address(address)
    components = parse_address_components(normalized)
    tokens = extract_tokens(normalized, min_length=config.get('address_token_min_length', 3),
                           stopwords=ADDRESS_STOPWORDS)
    
    # Postal code prefix
    if config.get('use_postal_code_prefix', True) and components['postal_code']:
        prefix_len = config.get('postal_code_prefix_length', 3)
        keys.append(f"postal_prefix:{components['postal_code'][:prefix_len]}")
        keys.append(f"postal_full:{components['postal_code']}")
    
    # House number
    if components['house_number'] and config.get('use_house_number', True):
        keys.append(f"house_num:{components['house_number']}")
    
    # Street name tokens
    street_tokens = extract_tokens(components['street'], min_length=3, stopwords=ADDRESS_STOPWORDS)
    for token in street_tokens[:3]:
        keys.append(f"street_token:{token}")
    
    # City
    if components['city'] and config.get('use_city', True):
        city_norm = normalize_name(components['city'])
        keys.append(f"city:{city_norm}")
    
    # State
    if components['state'] and config.get('use_state', True):
        keys.append(f"state:{components['state'].lower()}")
    
    # N-grams on full address
    ngram_size = config.get('address_ngram_size', 3)
    if config.get('use_ngrams', True):
        ngrams = extract_ngrams(normalized.replace(' ', ''), n=ngram_size)
        for ng in ngrams[:10]:
            keys.append(f"addr_ngram:{ng}")
    
    return keys


# Phonetic algorithms
def soundex(text: str) -> str:
    """Generate Soundex code."""
    if not text:
        return '0000'
    
    text = text.upper()
    first_char = text[0]
    
    # Mapping
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    # Encode
    encoded = first_char
    prev_code = mapping.get(first_char, '')
    
    for char in text[1:]:
        code = mapping.get(char, '')
        if code and code != prev_code:
            encoded += code
        prev_code = code if code else prev_code
    
    # Pad or truncate to 4 chars
    encoded = (encoded + '0000')[:4]
    return encoded


def metaphone(text: str) -> str:
    """Simple Metaphone implementation."""
    if not text:
        return ''
    
    text = text.upper()
    result = []
    i = 0
    
    # Handle initial letters
    if text.startswith('KN') or text.startswith('GN') or text.startswith('PN') or \
       text.startswith('AE') or text.startswith('WR'):
        i = 1
    elif text.startswith('X'):
        result.append('S')
        i = 1
    elif text.startswith('WH'):
        result.append('W')
        i = 1
    
    while i < len(text):
        char = text[i]
        
        # Skip duplicates except C
        if i > 0 and char == text[i-1] and char != 'C':
            i += 1
            continue
        
        if char == 'A' or char == 'E' or char == 'I' or char == 'O' or char == 'U' or char == 'Y':
            if i == 0:
                result.append(char)
        elif char == 'B':
            if i == len(text) - 1 and text[i-1] == 'M':
                pass
            else:
                result.append('B')
        elif char == 'C':
            if i+1 < len(text):
                if text[i+1] in 'EIY':
                    result.append('S')
                elif text[i+1] == 'H':
                    result.append('X')
                    i += 1
                else:
                    result.append('K')
            else:
                result.append('K')
        elif char == 'D':
            if i+1 < len(text) and text[i+1] in 'GE':
                result.append('J')
                i += 1
            else:
                result.append('T')
        elif char == 'F':
            result.append('F')
        elif char == 'G':
            if i+1 < len(text):
                if text[i+1] in 'EIY':
                    result.append('J')
                elif text[i+1] == 'H' and i+2 < len(text) and text[i+2] not in 'AEIOU':
                    result.append('K')
                    i += 1
                elif text[i+1] == 'N' and i == len(text) - 2:
                    pass
                else:
                    result.append('K')
            else:
                result.append('K')
        elif char == 'H':
            if i == 0 or (i > 0 and text[i-1] not in 'AEIOU'):
                if i+1 < len(text) and text[i+1] in 'AEIOU':
                    result.append('H')
        elif char == 'J':
            result.append('J')
        elif char == 'K':
            if i == 0 or text[i-1] != 'C':
                result.append('K')
        elif char == 'L':
            result.append('L')
        elif char == 'M':
            result.append('M')
        elif char == 'N':
            result.append('N')
        elif char == 'P':
            if i+1 < len(text) and text[i+1] == 'H':
                result.append('F')
                i += 1
            else:
                result.append('P')
        elif char == 'Q':
            result.append('K')
        elif char == 'R':
            result.append('R')
        elif char == 'S':
            if i+1 < len(text) and text[i+1] in 'CH':
                result.append('X')
                i += 1
            else:
                result.append('S')
        elif char == 'T':
            if i+1 < len(text) and text[i+1] == 'I' and i+2 < len(text) and text[i+2] in 'AO':
                result.append('X')
                i += 1
            elif i+1 < len(text) and text[i+1] == 'H':
                result.append('0')
                i += 1
            else:
                result.append('T')
        elif char == 'V':
            result.append('F')
        elif char == 'W':
            if i+1 < len(text) and text[i+1] in 'AEIOU':
                result.append('W')
        elif char == 'X':
            result.append('KS')
        elif char == 'Z':
            result.append('S')
        
        i += 1
    
    return ''.join(result)[:10]


def nysiis(text: str) -> str:
    """NYSIIS phonetic algorithm."""
    if not text:
        return ''
    
    text = text.upper()
    # Simplified NYSIIS
    # Replace prefixes
    for prefix, repl in [('MAC', 'MCC'), ('KN', 'N'), ('K', 'C'), ('PH', 'FF'), ('PF', 'FF'), ('SCH', 'SSS')]:
        if text.startswith(prefix):
            text = repl + text[len(prefix):]
            break
    
    # Replace suffixes
    for suffix, repl in [('EE', 'Y'), ('IE', 'Y'), ('DT', 'D'), ('RT', 'RD'), ('RD', 'RD'), ('NT', 'ND'), ('ND', 'ND')]:
        if text.endswith(suffix):
            text = text[:-len(suffix)] + repl
            break
    
    # First letter
    result = text[0]
    
    # Process rest
    i = 1
    while i < len(text):
        char = text[i]
        prev = result[-1] if result else ''
        
        if char in 'AEIOU':
            if prev != 'A':
                result += 'A'
        elif char == 'Q':
            result += 'G'
        elif char == 'Z':
            result += 'S'
        elif char == 'M':
            result += 'N'
        elif char == 'K':
            if prev == 'C':
                result += 'C'
            else:
                result += 'K'
        elif char == 'S':
            result += 'S'
        elif char == 'P':
            if prev == 'H':
                result = result[:-1] + 'F'
            else:
                result += 'P'
        elif char == 'H':
            if prev not in 'AEIOU' and (i+1 < len(text) and text[i+1] in 'AEIOU'):
                result += prev
        elif char == 'W':
            if prev in 'AEIOU':
                result += 'W'
        else:
            if char != prev:
                result += char
        
        i += 1
    
    # Remove trailing S
    if result.endswith('S'):
        result = result[:-1]
    
    # Remove trailing A
    if result.endswith('A'):
        result = result[:-1]
    
    return result[:6]


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def token_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token overlap ratio."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / min(len(set1), len(set2))