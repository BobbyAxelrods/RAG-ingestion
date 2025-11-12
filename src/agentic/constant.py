"""Shared constants for PIL Document Search API

This module contains constants used across multiple modules to ensure
consistency and maintain a single source of truth.
"""

# Comprehensive PIL product name mapping (English + Chinese)
# Format: (English_pattern, Chinese_pattern) tuples for regex matching
PRODUCT_NAMES = {
    # PRUwealth Series
    ('PRUwealth', '盈利寶'),
    # PRUChoice Healthcare Products
    ('PRUChoice Clinic', '保誠精選.*診療寶|診療寶'),
    ('PRUChoice HealthCare|PRUchoice HealthCare', '保誠精選.*康療寶|康療寶'),
    ('PRUChoice Medical|PRUchoice Medical', '保誠精選.*醫療寶|醫療寶'),
    ('PRUchoice MediExtra', '保誠精選.*健康寶|健康寶'),
    ('PRUchoice HealthCheck', '保誠精選.*康檢寶|康檢寶'),
    # PRUChoice Accident Products
    ('PRUChoice Personal Accident|PRUchoice Personal Accident', '保誠精選.*安健寶|安健寶'),
    ('PRUChoice Personal Accident Plus|PRUchoice Personal Accident Plus', '保誠精選.*倍安寶|倍安寶'),
    # PRUChoice Travel Products
    ('PRUChoice Travel|PRUchoice Travel', '保誠精選.*旅遊樂|旅遊樂'),
    ('PRUchoice Cruise Travel', '保誠精選.*郵輪旅遊樂|郵輪旅遊樂'),
    ('Working Holiday Insurance Plan', '保誠精選.*工作假期寶|工作假期寶'),
    ('Overseas Study Insurance Plan', '保誠精選.*海外留學寶|海外留學寶'),
    ('PRUChoice Hong Kong Study Care', '保誠精選.*來港尚學寶|來港尚學寶'),
    ('PRUChoice Relocation Care', '保誠精選.*移居寶|移居寶'),
    # PRUChoice Property Products
    ('PRUChoice Motor|PRUchoice Motor', '保誠精選.*駕駛寶|駕駛寶'),
    ('PRUChoice Home|PRUchoice Home', '保誠精選.*家居寶|家居寶'),
    ('PRUChoice Home Deluxe|PRUchoice Home Deluxe', '保誠精選.*名家寶|名家寶'),
    ('PRUChoice Home Décor', '保誠精選.*家居裝修寶|家居裝修寶'),
    ('PRUChoice Home Landlord|PRUchoice Home Landlord', '保誠精選.*業主寶|業主寶'),
    ('PRUChoice Office', '保誠精選.*興業寶|興業寶'),
    ('PRUChoice Shop', '保誠精選.*商舖寶|商舖寶'),
    # PRUChoice Lifestyle Products
    ('PRUChoice Maid|PRUchoice Maid', '保誠精選.*僱傭寶|僱傭寶'),
    ('PRUChoice Furkid Care', '保誠精選.*寵愛寶|寵愛寶'),
    ('PRUchoice Card Protection Plus', '保誠精選.*失卡寶|失卡寶'),
    # PRUChoice China Products
    ('PRUchoice China Protection', '保誠精選.*中國安心寶|中國安心寶'),
    ('PRUchoice China Accidental Emergency Medical', '保誠精選.*中國意外急救|中國意外急救'),
    # PRUChoice Sports Products
    ('PRUchoice Golfers', '保誠精選.*高球樂|高球樂'),
    # Generic product categories (broader matching)
    ('Travel Insurance|Travel', '旅遊保險|旅游保險'),
    ('Motor Insurance|Motor', '汽車保險|車輛保險'),
    ('Health Insurance|Healthcare', '健康保險|醫療保險'),
    ('Maid Insurance|Domestic Helper', '家傭保險'),
    ('Personal Accident', '意外保險'),
    ('Critical Illness', '危疾保險'),
    ('Hospital', '住院保險'),
    # Investment-linked Assurance Scheme
    ('PRULink Opal Investment Plan','雋珀投資計劃'),
    # Legacy Planning | Life insurance
    ('Prime Vantage Protector','雍譽終身壽險計劃')
}
