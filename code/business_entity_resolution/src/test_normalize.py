"""
Unit tests for normalize.py.
Verifies normalization rules, abbreviations, and landmark/postal code extraction.
"""

from normalize import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)


def test_business_name_normalization():
    # Legal mappings
    name1, tokens1, acr1, comp1 = normalize_business_name("Acme Corporation & Co.")
    name2, tokens2, acr2, comp2 = normalize_business_name("Acme Corp and Company")
    assert name1 == "acme corp and co"
    assert name2 == "acme corp and co"
    assert tokens1 == tokens2
    assert comp1 == comp2 == "acmecorpandco"

    # Acronym
    _, _, acr, _ = normalize_business_name("International Business Machines")
    assert acr == "ibm"

    # Private Limited
    name3, _, _, _ = normalize_business_name("Reliance Industries Private Limited")
    assert name3 == "reliance ind pvt ltd"

    # Domain name
    name4, tokens4, _, comp4 = normalize_business_name("maurewilliamscolombier.com")
    assert name4 == "maurewilliamscolombier"
    assert comp4 == "maurewilliamscolombier"


def test_business_address_normalization():
    # Address abbreviations and street number
    addr1, tokens1, pin1, st_num1, landmark1 = normalize_business_address("123 Main Street, Suite 400, Rd 5")
    assert "st" in tokens1
    assert "rd" in tokens1
    assert st_num1 == "123"

    # Landmark and PIN extraction
    addr2, tokens2, pin2, st_num2, landmark2 = normalize_business_address(
        "Near SBI ATM, MG Road, Bangalore 560001"
    )
    assert landmark2 == "sbi atm"
    assert pin2 == "560001"


def test_country_normalization():
    assert normalize_country("US") == "US"
    assert normalize_country("USA") == "US"
    assert normalize_country("India") == "INDIA"
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("Germany") == "GERMANY"  # open set, doesn't drop
