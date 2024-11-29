import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import json

client = OpenAI()

META_SCHEMA = {
  "name": "metaschema",
  "schema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "The name of the schema"
      },
      "type": {
        "type": "string",
        "enum": [
          "object",
          "array",
          "string",
          "number",
          "boolean",
          "null"
        ]
      },
      "properties": {
        "type": "object",
        "additionalProperties": {
          "$ref": "#/$defs/schema_definition"
        }
      },
      "items": {
        "anyOf": [
          {
            "$ref": "#/$defs/schema_definition"
          },
          {
            "type": "array",
            "items": {
              "$ref": "#/$defs/schema_definition"
            }
          }
        ]
      },
      "required": {
        "type": "array",
        "items": {
          "type": "string"
        }
      },
      "additionalProperties": {
        "type": "boolean"
      }
    },
    "required": [
      "type"
    ],
    "additionalProperties": False,
    "if": {
      "properties": {
        "type": {
          "const": "object"
        }
      }
    },
    "then": {
      "required": [
        "properties"
      ]
    },
    "$defs": {
      "schema_definition": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "object",
              "array",
              "string",
              "number",
              "boolean",
              "null"
            ]
          },
          "properties": {
            "type": "object",
            "additionalProperties": {
              "$ref": "#/$defs/schema_definition"
            }
          },
          "items": {
            "anyOf": [
              {
                "$ref": "#/$defs/schema_definition"
              },
              {
                "type": "array",
                "items": {
                  "$ref": "#/$defs/schema_definition"
                }
              }
            ]
          },
          "required": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "additionalProperties": {
            "type": "boolean"
          }
        },
        "required": [
          "type"
        ],
        "additionalProperties": False,
        "if": {
          "properties": {
            "type": {
              "const": "object"
            }
          }
        },
        "then": {
          "required": [
            "properties"
          ]
        }
      }
    }
  }
}

META_PROMPT = """
# Instructions
Return a valid schema for the described JSON.

You must also make sure:
- all fields in an object are set as required
- I REPEAT, ALL FIELDS MUST BE MARKED AS REQUIRED
- all objects must have additionalProperties set to false
    - because of this, some cases like "attributes" or "metadata" properties that would normally allow additional properties should instead have a fixed set of properties
- all objects must have properties defined
- field order matters. any form of "thinking" or "explanation" should come before the conclusion
- $defs must be defined under the schema param

Notable keywords NOT supported include:
- For strings: minLength, maxLength, pattern, format
- For numbers: minimum, maximum, multipleOf
- For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
- For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems

Other notes:
- definitions and recursion are supported
- only if necessary to include references e.g. "$defs", it must be inside the "schema" object

# Examples
Input: Generate a math reasoning schema with steps and a final answer.
Output: {
    "name": "math_reasoning",
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "A sequence of steps involved in solving the math problem.",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Description of the reasoning or method used in this step."
                    },
                    "output": {
                        "type": "string",
                        "description": "Result or outcome of this specific step."
                    }
                },
                "required": [
                    "explanation",
                    "output"
                ],
                "additionalProperties": false
            }
        },
        "final_answer": {
            "type": "string",
            "description": "The final solution or answer to the math problem."
        }
    },
    "required": [
        "steps",
        "final_answer"
    ],
    "additionalProperties": false
}

Input: Give me a linked list
Output: {
    "name": "linked_list",
    "type": "object",
    "properties": {
        "linked_list": {
            "$ref": "#/$defs/linked_list_node",
            "description": "The head node of the linked list."
        }
    },
    "$defs": {
        "linked_list_node": {
            "type": "object",
            "description": "Defines a node in a singly linked list.",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The value stored in this node."
                },
                "next": {
                    "anyOf": [
                        {
                            "$ref": "#/$defs/linked_list_node"
                        },
                        {
                            "type": "null"
                        }
                    ],
                    "description": "Reference to the next node; null if it is the last node."
                }
            },
            "required": [
                "value",
                "next"
            ],
            "additionalProperties": false
        }
    },
    "required": [
        "linked_list"
    ],
    "additionalProperties": false
}

Input: Dynamically generated UI
Output: {
    "name": "ui",
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "description": "The type of the UI component",
            "enum": [
                "div",
                "button",
                "header",
                "section",
                "field",
                "form"
            ]
        },
        "label": {
            "type": "string",
            "description": "The label of the UI component, used for buttons or form fields"
        },
        "children": {
            "type": "array",
            "description": "Nested UI components",
            "items": {
                "$ref": "#"
            }
        },
        "attributes": {
            "type": "array",
            "description": "Arbitrary attributes for the UI component, suitable for any element",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the attribute, for example onClick or className"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value of the attribute"
                    }
                },
                "required": [
                    "name",
                    "value"
                ],
                "additionalProperties": false
            }
        }
    },
    "required": [
        "type",
        "label",
        "children",
        "attributes"
    ],
    "additionalProperties": false
}
""".strip()

def generate_schema(description: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_schema", "json_schema": META_SCHEMA},
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": "Description:\n" + description,
            },
        ],
    )

    return json.loads(completion.choices[0].message.content)

if __name__ == "__main__":
    print(generate_schema("""{
        "resume_analysis": {
          "personal_info": {
            "name": "Full name",
            "contact": "Dictionary of contact information",
            "social_profiles": "Dictionary of social profiles",
            "...": "..."
          },
          "professional_summary": "Paragraph of professional summary",
          "work_experience": [ "List of work experiences with the following structure",
            {
              "work_experience": "Paragraph of work experience",
              "company": {
                "name": "Company name",
                "industry": "Industry type",
                "location": "Company location",
                "...": "..."
              },
              "position": "Job title",
              "duration": {
                "start_date": "YYYY-MM",
                "end_date": "YYYY-MM or Present",
                "total_duration": "Duration in years/months"
              }
            },
            "..."
          ],
          "skills": "Skills including technical skills, soft skills, tools, etc.",
          "languages": [
            {
              "language": "Language name",
              "proficiency": "Proficiency level",
            },
            "..."
          ],
          "education": [
            {
              "degree": "Degree name",
              "field": "Field of study",
              "institution": "Institution name",
              "location": "Institution location",
              "duration": {
                "start_date": "YYYY-MM",
                "end_date": "YYYY-MM",
                "graduation_status": "Completed/Ongoing"
              },
              "gpa": {
                "score": "GPA value",
                "scale": "GPA scale (e.g. 4.0, 10.0)",
                "percentage": "Percentage equivalent (optional)"
              }
            },
            "..."
          ],
          "certifications": [
            {
              "name": "Certification name",
              "issuer": "Issuing organization",
              "date_obtained": "YYYY-MM",
              "expiry_date": "YYYY-MM or Never",
              "credential_id": "Certification ID",
            },
            "..."
          ],
          "projects": [
            {
              "name": "Project name",
              "duration": {
                "start_date": "YYYY-MM",
                "end_date": "YYYY-MM",
                "total_duration": "Duration in months"
              },
              "description": "Project description in bullet points or a paragraph",
            },
            "..."
          ],
          "research_and_publications": [
            {
              "title": "Publication/Research title",
              "description": "Publication/Research description in bullet points or a paragraph",
            },
            "..."
          ],
          "awards_and_achievements": [
            {
              "title": "Award/Achievement title",
              "description": "Award/Achievement description in bullet points or a paragraph",
            },
            "..."
          ],
          "volunteer_work": [
            {
              "organization": "Organization name",
              "role": "Role description",
              "duration": "Duration in months"
            },
            "..."
          ],
          "positions_of_responsibility": [
            {
              "title": "Position title",
              "organization": "Organization name",
              "duration": "Duration in months"
            },
            "..."
          ],
          "ambiguous_content": [
            {
              "section": "Section name",
              "content": "Unclear content",
              "issue": "Brief description of ambiguity",
            },
            "..."
          ],
          "...": "..."
        }
      }
"""))