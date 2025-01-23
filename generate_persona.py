"""
Generate diverse personas for authorship attribution defense studies.

This module creates realistic personas by combining:
1. Demographic information (via Faker)
2. Personality traits (via Big Five Inventory-10)
3. Professional backgrounds (via O*NET 29.1 Database)

The BFI-10 is a short version of the Big Five Inventory that measures
personality dimensions using just 10 items, making it efficient while
maintaining acceptable psychometric properties.

BFI-10 Dimensions:
- Extraversion: outgoing/reserved (items 1R, 6)
- Agreeableness: trusting/critical (items 2, 7R)
- Conscientiousness: thorough/careless (items 3, 8R)
- Neuroticism: relaxed/nervous (items 4R, 9)
- Openness: imaginative/conventional (items 5, 10R)
(R = reverse scored items)

Output Structure:
    prompts/rq3.1_persona_playing/
    ├── persona_000.json  # individual persona prompts
    ├── persona_001.json
    ├── ...
    └── persona_999.json

Each persona file contains:
    {
        "system": str,      # system prompt with persona description
        "user": str,        # user prompt template
        "metadata": dict    # full persona details including BFI-10 scores
    }

Usage:
    # Generate 1000 persona prompts
    python persona_generator.py

Dependencies:
    - pandas: for reading O*NET data
    - faker: for generating demographic information
    - pathlib: for path handling

Source:
    BFI-10: https://socialwork.buffalo.edu/content/dam/socialwork/home/self-care-kit/brief-big-five-personality-inventory.pdf
    O*NET 29.1 Database: https://www.onetcenter.org/database.html
"""

import random
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from faker import Faker
from dataclasses import dataclass


# initialize faker with consistent seed
fake = Faker()
Faker.seed(42)


@dataclass
class PersonalityTrait:
    """representation of an IPIP-NEO personality trait"""
    domain: str
    facet: str
    description: str
    score: int  # 1-5 scale


class PersonaGenerator:
    """generates comprehensive personas from multiple data sources"""

    def __init__(
        self,
        onet_path: str = "/resources/db_29_1_excel/Occupation Data.xlsx",
        seed: int = 42
    ):
        """initialize with data sources"""
        random.seed(seed)
        self.occupations = self._load_onet_data(onet_path)
        self.personality_traits = self._load_personality_traits()

    def _load_onet_data(self, file_path: str) -> pd.DataFrame:
        """load O*NET occupational data"""
        df = pd.read_excel(file_path)
        required_cols = ['O*NET-SOC Code', 'Title', 'Description']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Excel file must contain columns: {required_cols}")
        return df

    def _load_personality_traits(self) -> Dict[str, List[str]]:
        """
        load IPIP-NEO traits (placeholder - replace with actual IPIP data)
        """
        return {
            "openness": [
                "curious about many different things",
                "inventive and creative",
                "values artistic experiences",
                "likes to reflect on things",
                "sophisticated in arts and music"
            ],
            "conscientiousness": [
                "does tasks thoroughly",
                "makes plans and follows through",
                "organized and methodical",
                "reliable in commitments",
                "persistent in tasks"
            ],
            "extraversion": [
                "outgoing and sociable",
                "energetic in approach",
                "confident in social situations",
                "takes charge naturally",
                "expressive in communication"
            ],
            "agreeableness": [
                "considerate of others",
                "cooperative in groups",
                "trusting of people",
                "helpful to others",
                "harmonious in relationships"
            ],
            "neuroticism": [
                "handles stress well",
                "emotionally stable",
                "adapts to change easily",
                "stays calm under pressure",
                "confident in abilities"
            ]
        }

    def _generate_demographics(self) -> Dict:
        """generate consistent demographic details"""
        gender = random.choice(['Male', 'Female'])

        return {
            'name': fake.name_male() if gender == 'Male' else fake.name_female(),
            'gender': gender,
            'age': random.randint(25, 65),
            'education': random.choice([
                "Bachelor's Degree",
                "Master's Degree",
                'Ph.D.',
                'Professional Degree'
            ]),
            'location': fake.city() + ', ' + fake.state(),
            'email': fake.email(),
            'background': fake.text(max_nb_chars=200)
        }

    def _generate_personality(self) -> Dict[str, List[Dict]]:
        """generate personality profile using IPIP-NEO framework"""
        profile = {}
        for domain, facets in self.personality_traits.items():
            traits = []
            num_traits = random.randint(2, 3)
            selected_facets = random.sample(facets, num_traits)

            for facet in selected_facets:
                score = random.randint(4, 5)  # high expression of selected traits
                trait = PersonalityTrait(
                    domain=domain,
                    facet="",  # placeholder
                    description=facet,
                    score=score
                )
                traits.append(trait.__dict__)  # convert to dictionary
            profile[domain] = traits
        return profile

    def _generate_occupation(self) -> Dict:
        """select random O*NET occupation"""
        row = self.occupations.sample(n=1).iloc[0]
        return {
            'title': row['Title'],
            'description': row['Description'],
            'code': row['O*NET-SOC Code']
        }

    def _format_personality_narrative(self, profile: Dict) -> str:
        """Format personality traits into a natural narrative"""
        narratives = []
        for domain, traits in profile.items():
            high_intensity_traits = [t for t in traits if t['score'] >= 4]
            if not high_intensity_traits:
                continue

            trait_descriptions = [t['description'] for t in high_intensity_traits]
            if len(trait_descriptions) == 1:
                narrative = trait_descriptions[0]
            elif len(trait_descriptions) == 2:
                narrative = f"{trait_descriptions[0]} and {trait_descriptions[1]}"
            else:
                narrative = f"{', '.join(trait_descriptions[:-1])}, and {trait_descriptions[-1]}"

            if domain == "openness":
                narratives.append(f"You approach tasks with {narrative}")
            elif domain == "conscientiousness":
                narratives.append(f"you are known for being {narrative}")
            elif domain == "extraversion":
                narratives.append(f"in your interactions, you are {narrative}")
            elif domain == "agreeableness":
                narratives.append(f"when working with others, you are {narrative}")
            elif domain == "neuroticism":
                narratives.append(f"under pressure, you are {narrative}")

        return ". ".join(narratives) + "."

    def generate_persona(self) -> Dict:
        """generate a complete persona"""
        demographics = self._generate_demographics()
        personality = self._generate_personality()
        occupation = self._generate_occupation()

        writing_style = (
            f"Their writing style reflects their role as a {occupation['title']} and their "
            f"personality traits. {self._format_personality_text(personality)}"
        )

        return {
            "demographics": demographics,
            "personality": {
                "profile": personality,
                "writing_style": writing_style
            },
            "occupation": occupation
        }

    def generate_prompt(self, persona: Dict) -> Dict[str, str]:
        """Create a naturally flowing prompt from a persona"""
        demographics = persona['demographics']
        occupation = persona['occupation']
        education = demographics.get('education', 'an unspecified level of education')

        # format work experience and education
        professional_background = (
            f"As a {occupation['title']} with {education}, your work involves "
            f"{occupation['description'].lower()}"
        )

        # create personality narrative
        personality_narrative = self._format_personality_narrative(
            persona['personality']['profile']
        )

        system_prompt = (
            f"You are {demographics['name']}, a {demographics['age']}-year-old professional from "
            f"{demographics['location']}. {professional_background}\n\n"
            f"{personality_narrative}\n\n"
            "Drawing from your background and personality, your task is to rewrite text while "
            "preserving its core meaning but expressing it through your unique perspective. "
            "Always enclose your rewritten version between [REWRITE] and [/REWRITE] tags."
        )

        user_prompt = (
            "Please rewrite the following text through your unique voice and perspective, "
            f"while maintaining its core meaning:\n\n{{{{text}}}}\n\n"
            "Provide your rewritten version between [REWRITE] and [/REWRITE] tags."
        )

        return {
            "system": system_prompt,
            "user": user_prompt,
            "metadata": persona
        }


def generate_persona_prompts(n: int = 1000, output_dir: str = "prompts/rq3.1_persona_playing") -> List[Dict]:
    """generate a set of persona-based prompts"""
    generator = PersonaGenerator()
    prompts = []

    # create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        # generate persona
        persona = generator.generate_persona()
        prompt = generator.generate_prompt(persona)

        # format prompt dict according to standard structure
        formatted_prompt = {
            "system": prompt["system"],
            "user": prompt["user"],
            "metadata": prompt["metadata"]
        }
        prompts.append(formatted_prompt)

        # save individual prompt with zero-padded numbering
        file_name = f"persona_{i:03d}.json"
        with open(output_path / file_name, 'w', encoding='utf-8') as f:
            json.dump(formatted_prompt, f, indent=4, ensure_ascii=False)

    return prompts


if __name__ == "__main__":
    # generate 1000 persona prompts
    prompts = generate_persona_prompts()
    print(f"Generated {len(prompts)} persona prompts in prompts/rq3.1_persona_playing/")
    print("\nExample prompt structure:")
    print(json.dumps(prompts[0], indent=2))
