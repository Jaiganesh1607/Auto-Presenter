from pydantic import BaseModel
from typing import Dict, List

class SectionSpec(BaseModel):
    id: str
    display_title: str
    scope_description: str

GENRE_TEMPLATES: Dict[str, List[SectionSpec]] = {
    "explainer": [
        SectionSpec(id="hook_title", display_title="Introduction", scope_description="An engaging opening paragraph that hooks the reader, clearly states the topic, and explains why this subject matters today."),
        SectionSpec(id="core_definition", display_title="Core Definition", scope_description="A precise, foundational definition of the topic that establishes exactly what it is, avoiding jargon where possible to ensure clear understanding."),
        SectionSpec(id="history_context", display_title="History and Context", scope_description="A chronological summary of the origins, key milestones, and major historical events that shaped this topic into its current state."),
        SectionSpec(id="types_categories", display_title="Types and Categories", scope_description="A structured breakdown of the distinct variations, classifications, or sub-genres that exist within this broader topic, explaining how they differ."),
        SectionSpec(id="mechanism_how_it_works", display_title="How It Works", scope_description="A step-by-step causal explanation of the physical, logical, or theoretical processes that enable the topic to function."),
        SectionSpec(id="real_world_applications", display_title="Real-World Applications", scope_description="Specific, documented examples of how this topic is currently being used by industries, organizations, or individuals in practical scenarios."),
        SectionSpec(id="benefits", display_title="Benefits", scope_description="A list of the primary advantages, positive impacts, and value propositions this topic offers to its users or society."),
        SectionSpec(id="challenges_risks", display_title="Challenges and Risks", scope_description="An objective analysis of the known limitations, ethical concerns, potential dangers, and common failure modes associated with this topic."),
        SectionSpec(id="future_outlook", display_title="Future Outlook", scope_description="Expert predictions and emerging trends detailing how this topic is expected to evolve over the next 5 to 10 years."),
        SectionSpec(id="synthesis_takeaways", display_title="Key Takeaways", scope_description="A concise final summary highlighting the 3-4 most critical insights the reader should remember about the topic."),
    ],
    "technical_how_it_works": [
        SectionSpec(id="hook_title", display_title="Introduction", scope_description="A technical hook that introduces the system or concept, clearly stating its primary function and target audience of engineers or developers."),
        SectionSpec(id="the_problem", display_title="The Problem", scope_description="the specific pain point, limitation, or failure mode that motivated this technology/concept's creation — search for how practitioners describe the problem in their own words, not an abstract framing"),
        SectionSpec(id="the_solution_concept", display_title="The Solution Concept", scope_description="A high-level architectural overview describing the primary components of the proposed solution and how they fit together to solve the problem."),
        SectionSpec(id="architecture_comparison", display_title="Architecture Comparison", scope_description="a direct comparison between 2-3 named architectural approaches for this domain, covering their tradeoffs — not a description of a single approach in isolation"),
        SectionSpec(id="core_vocabulary", display_title="Core Vocabulary", scope_description="Dictionary definitions of individual technical words, acronyms, and jargon. This section only contains short definitions of terminology, not system overviews."),
        SectionSpec(id="step_by_step_mechanism", display_title="Step-by-Step Mechanism", scope_description="A detailed, sequential breakdown tracing the exact execution path, data flow, or lifecycle from input to output."),
        SectionSpec(id="under_the_hood", display_title="Under the Hood", scope_description="A highly technical deep dive into the specific internal algorithms, data structures, protocols, and performance optimizations that power the system."),
        SectionSpec(id="summary_cheat_sheet", display_title="Summary Cheat Sheet", scope_description="A quick-reference technical summary listing key commands, essential configurations, or core design principles."),
    ],
    "business_market": [
        SectionSpec(id="hook_title", display_title="Executive Summary", scope_description="An executive summary that introduces the market or business topic, outlining the current macro environment and the main thesis of the analysis."),
        SectionSpec(id="market_landscape", display_title="Market Landscape", scope_description="Quantitative and qualitative description of the current state of the market, including total size, major segments, and overall maturity."),
        SectionSpec(id="key_players", display_title="Key Players", scope_description="A profile of the dominant companies, rising startups, and major stakeholders currently operating in this space, including their market share."),
        SectionSpec(id="opportunity_sizing", display_title="Opportunity Sizing", scope_description="An analysis of the Total Addressable Market (TAM), projected CAGR, and specific areas of untapped revenue or growth potential."),
        SectionSpec(id="competitive_dynamics", display_title="Competitive Dynamics", scope_description="An evaluation of how market players compete, detailing their moats, pricing strategies, and differentiation tactics."),
        SectionSpec(id="risks_headwinds", display_title="Risks and Headwinds", scope_description="An assessment of the regulatory, economic, technological, or supply-chain threats that could negatively impact market growth."),
        SectionSpec(id="outlook", display_title="Market Outlook", scope_description="Strategic recommendations and future projections predicting consolidation, disruption, or major shifts in the market landscape."),
    ],
    "comparison": [
        SectionSpec(id="hook_title", display_title="Introduction", scope_description="An engaging opening that introduces the two or more subjects being compared and establishes why a choice between them is necessary."),
        SectionSpec(id="framing_the_choice", display_title="Framing the Choice", scope_description="An explanation of the specific use cases, constraints, and criteria that make this comparison relevant to the decision-maker."),
        SectionSpec(id="option_a_deep_dive", display_title="Deep Dive: Option A", scope_description="A comprehensive, standalone analysis of the first option, detailing its core strengths, weaknesses, and ideal use cases."),
        SectionSpec(id="option_b_deep_dive", display_title="Deep Dive: Option B", scope_description="A comprehensive, standalone analysis of the second option, detailing its core strengths, weaknesses, and ideal use cases."),
        SectionSpec(id="head_to_head_criteria", display_title="Head-to-Head Criteria", scope_description="A direct, feature-by-feature comparison evaluating how both options stack up against each other across key dimensions like cost, performance, and usability."),
        SectionSpec(id="recommendation_synthesis", display_title="Recommendation & Synthesis", scope_description="A final verdict providing clear, situational recommendations on which option to choose based on specific user needs."),
    ]
}
