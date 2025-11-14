from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional


class EntityType(str, Enum):
    PRODUCT = "product"      # Insurance products: PRUactive, PRUhealth, etc.
    PEOPLE = "people"        # Policyowner, beneficiary, agent, etc.
    ACTION = "action"        # claim, payment, cancel, renew, etc.
    METHOD = "method"        # online, autopay, bank transfer, etc.
    CATEGORY = "category"    # cashier, claims, forms, etc.

class QueryIntent(str, Enum):
    """Query intent types"""
    INFORMATIONAL_QUERY = "informational_query"     # "What is...?"
    HOW_TO_QUERY = "how_to_query"                   # "How do I...?"
    DOCUMENT_LOOKUP = "document_lookup"             # "Find form CA000001"
    COMPARISON_QUERY = "comparison_query"           # "Compare X and Y"
    TROUBLESHOOTING = "troubleshooting"             # "Why can't I...?"
    DEFINITION_QUERY = "definition_query"           # "What does X mean?"
    PROCESS_QUERY = "process_query"                 # "What is the process for...?"


class QueryAnalysisOutput(BaseModel):
    product_entities : List[str] = Field(
        description= "Insurance product names mentioned (e.g PRUACTIVE, PRUHEALTH etc)"
    )