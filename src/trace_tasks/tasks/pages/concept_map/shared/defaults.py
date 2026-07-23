"""Defaults, palettes, and text resources for concept-map scene packages."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.diagram.visual_defaults import load_diagrams_background_defaults, load_diagrams_noise_defaults
from trace_tasks.tasks.pages.shared.page_semantic_assets import (
    page_semantic_asset_ids,
    page_semantic_asset_label,
)
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


DOMAIN = "pages"
SCENE = "concept_map"
PROMPT_BUNDLE = "pages_concept_map_v1"
PROMPT_SCENE_KEY = "concept_map_diagram"
PROMPT_TASK_KEY = "concept_map_lookup_query"
NAMESPACE_ROOT = "pages.concept_map"

CHILDREN_TOTAL_KIND = "children_total"
MARKED_TOTAL_KIND = "marked_total"
RANKED_CHILD_KIND = "ranked_child"

LAYOUT_VARIANTS: Tuple[str, ...] = ("radial_mind_map", "left_right_map", "clustered_map")
STYLE_VARIANTS: Tuple[str, ...] = ("bright_notes", "ink_outline", "soft_cards", "technical_pastel")
NODE_SHAPE_PROFILES: Tuple[str, ...] = ("mixed_hub_circle", "oval_branch_mix", "mixed_cards_ovals")
CHILD_NODE_SHAPES: Tuple[str, ...] = ("rounded_rect", "ellipse", "pill")
CONTEXT_VARIANTS: Tuple[str, ...] = (
    "travel_plans",
    "career_options",
    "climate_actions",
    "community_groups",
    "shopping_tips",
    "science_topics",
)

MARKERS: Tuple[Dict[str, Any], ...] = tuple(
    {
        "marker_id": str(marker_id),
        "label": page_semantic_asset_label(str(marker_id)),
        "semantic_id": str(marker_id),
    }
    for marker_id in page_semantic_asset_ids(semantic_role="marker", allowed_use="filter")
)

PALETTES: Dict[str, Dict[str, Any]] = {
    "bright_notes": {
        "panel_fill": (255, 255, 250),
        "panel_outline": (190, 197, 212),
        "central_fill": (43, 59, 128),
        "central_outline": (21, 31, 78),
        "central_text": (255, 255, 255),
        "node_text": (33, 41, 58),
        "connector": (90, 102, 132),
        "branch_fills": (
            (255, 218, 104),
            (255, 149, 164),
            (136, 212, 151),
            (153, 191, 255),
            (218, 164, 255),
            (255, 185, 114),
            (126, 223, 215),
            (231, 226, 118),
        ),
    },
    "ink_outline": {
        "panel_fill": (248, 247, 241),
        "panel_outline": (72, 77, 88),
        "central_fill": (45, 48, 56),
        "central_outline": (20, 22, 28),
        "central_text": (255, 255, 250),
        "node_text": (32, 34, 38),
        "connector": (78, 82, 90),
        "branch_fills": (
            (245, 237, 193),
            (237, 210, 208),
            (209, 231, 214),
            (207, 224, 238),
            (229, 213, 236),
            (238, 221, 199),
            (207, 231, 228),
            (229, 228, 201),
        ),
    },
    "soft_cards": {
        "panel_fill": (245, 249, 252),
        "panel_outline": (174, 191, 204),
        "central_fill": (45, 94, 111),
        "central_outline": (24, 60, 74),
        "central_text": (255, 255, 255),
        "node_text": (31, 47, 58),
        "connector": (88, 126, 139),
        "branch_fills": (
            (188, 225, 232),
            (244, 207, 199),
            (208, 231, 197),
            (214, 218, 246),
            (246, 221, 183),
            (217, 205, 238),
            (194, 231, 216),
            (241, 220, 224),
        ),
    },
    "technical_pastel": {
        "panel_fill": (247, 248, 252),
        "panel_outline": (148, 158, 179),
        "central_fill": (63, 73, 105),
        "central_outline": (34, 41, 68),
        "central_text": (255, 255, 255),
        "node_text": (35, 42, 60),
        "connector": (88, 99, 130),
        "branch_fills": (
            (228, 235, 255),
            (255, 228, 230),
            (226, 244, 233),
            (245, 235, 255),
            (255, 238, 211),
            (224, 245, 246),
            (240, 236, 211),
            (233, 230, 248),
        ),
    },
}

CONTEXTS: Dict[str, Dict[str, Any]] = {
    "travel_plans": {
        "title": "Travel planning concept map",
        "central": "Travel Plans",
        "branches": {
            "Domestic": ["New York", "Denver", "Austin", "Seattle", "Miami", "Chicago", "Boston", "Phoenix"],
            "International": ["Paris", "Tokyo", "Lisbon", "Toronto", "Seoul", "Dublin", "Madrid", "Cairo"],
            "Tickets": ["Rail Pass", "Flight Hold", "Seat Map", "Boarding", "Upgrade", "Refund", "Transfer", "Voucher"],
            "Lodging": ["Hotel", "Hostel", "Cabin", "Apartment", "Resort", "Guesthouse", "Inn", "Suite"],
            "Activities": ["Museum", "Garden", "Boat Tour", "Market", "Hiking", "Theater", "Food Walk", "Stadium"],
            "Transport": ["Metro", "Taxi", "Rental Car", "Bike Share", "Ferry", "Shuttle", "Tram", "Bus Pass"],
            "Packing": ["Passport", "Charger", "Camera", "Jacket", "Snacks", "Adapter", "Notebook", "Umbrella"],
            "Budget": ["Meals", "Tickets", "Hotel Tax", "Tips", "Transit", "Insurance", "Tours", "Souvenirs"],
        },
    },
    "career_options": {
        "title": "Career transition concept map",
        "central": "Career Options",
        "branches": {
            "Media": ["Commentator", "Host", "Producer", "Editor", "Podcaster", "Reviewer", "Reporter", "Analyst"],
            "Coaching": ["Youth Coach", "Trainer", "Scout", "Mentor", "Playbook", "Camp Lead", "Skills Coach", "Tutor"],
            "Business": ["Agent", "Sponsor", "Founder", "Consultant", "Advisor", "Investor", "Manager", "Recruiter"],
            "Education": ["Teacher", "Lecturer", "Workshop", "Curriculum", "Seminar", "Coach Cert", "Tutor", "Course"],
            "Community": ["Volunteer", "Ambassador", "Fundraiser", "Outreach", "Board Seat", "Program Lead", "Clinic", "Mentor"],
            "Health": ["Therapist", "Nutrition", "Wellness", "Recovery", "Strength", "Mindset", "Balance", "Mobility"],
            "Creative": ["Author", "Designer", "Video", "Branding", "Photography", "Podcast", "Storytelling", "Studio"],
            "Operations": ["Planner", "Director", "Scheduler", "Coordinator", "Logistics", "Facilities", "Compliance", "Events"],
        },
    },
    "climate_actions": {
        "title": "Climate action concept map",
        "central": "Climate Action",
        "branches": {
            "Weather": ["Heat Wave", "Flood", "Storm", "Drought", "Smoke", "Wind", "Cold Snap", "Humidity"],
            "Energy": ["Solar", "Wind Power", "Storage", "Grid", "Retrofit", "Metering", "Backup", "Efficiency"],
            "Water": ["Rain Garden", "Reuse", "Drainage", "Reservoir", "Irrigation", "Leak Audit", "Filter", "Drought Plan"],
            "Transport": ["Carpool", "EV Bus", "Bike Lane", "Rail", "Walk Route", "Charging", "Shuttle", "Transit Card"],
            "Food": ["Compost", "Local Farm", "Cold Chain", "Menu Shift", "Food Bank", "Garden", "Storage", "Waste Log"],
            "Buildings": ["Insulation", "Cool Roof", "Shade", "Window Film", "Heat Pump", "Sensor", "Ventilation", "Audit"],
            "Policy": ["Permit", "Grant", "Code", "Target", "Report", "Dashboard", "Budget", "Review"],
            "Outreach": ["Workshop", "Newsletter", "Survey", "Hotline", "School Visit", "Poster", "Volunteer", "Briefing"],
        },
    },
    "community_groups": {
        "title": "Community organization concept map",
        "central": "Local Network",
        "branches": {
            "Nonprofits": ["Food Pantry", "Youth Arts", "Legal Aid", "Book Bank", "Shelter", "Green Team", "Free Clinic", "Music Fund"],
            "Schools": ["High School", "Art Club", "Library Lab", "STEM Camp", "Parent Board", "Tutoring", "Chess Club", "Drama Room"],
            "Health": ["Clinic", "Counseling", "Dental Van", "Nutrition", "Wellness Fair", "Blood Drive", "Care Line", "Pharmacy"],
            "Safety": ["Watch Team", "Fire Dept", "CERT", "Hotline", "Crossing Guard", "Shelter Map", "Alert Desk", "First Aid"],
            "Culture": ["Museum", "Choir", "Dance Class", "Film Night", "Theater", "Festival", "Gallery", "Poetry"],
            "Parks": ["Trail Crew", "Gardeners", "Tree Board", "Playground", "Dog Park", "Clean Up", "Picnic", "Bird Walk"],
            "Housing": ["Tenant Help", "Repair Crew", "Rent Clinic", "New Units", "Survey", "Co-op", "Shelter Link", "Mediation"],
            "Jobs": ["Resume Lab", "Apprentice", "Job Fair", "Career Desk", "Training", "Mentor Net", "Startup", "Internship"],
        },
    },
    "shopping_tips": {
        "title": "Shopping advice concept map",
        "central": "Shopping Tips",
        "branches": {
            "Authenticity": ["Serial Code", "Receipt", "Logo Check", "Seller Rating", "Material", "Packaging", "Warranty", "Photo Match"],
            "Budget": ["Coupon", "Price Alert", "Bundle", "Cashback", "Clearance", "Tax", "Shipping", "Return Fee"],
            "Quality": ["Stitching", "Weight", "Reviews", "Fit", "Durability", "Finish", "Color Fast", "Battery"],
            "Safety": ["Recall", "Age Label", "Seal", "Ingredient", "Voltage", "Allergy", "Certification", "Warning"],
            "Timing": ["Holiday", "Restock", "Preorder", "Season End", "Flash Sale", "Launch Day", "Weekend", "Closeout"],
            "Delivery": ["Pickup", "Tracking", "Locker", "Courier", "Signature", "Rush", "Packaging", "Delay"],
            "Sustainability": ["Repair", "Refill", "Used", "Rental", "Local", "Organic", "Recycled", "Low Waste"],
            "Comparison": ["Size Chart", "Feature List", "Warranty", "Unit Price", "Sample", "Trial", "Spec Sheet", "Benchmark"],
        },
    },
    "science_topics": {
        "title": "Science topic concept map",
        "central": "Science Topics",
        "branches": {
            "Astronomy": ["Leo", "Regulus", "Orion", "Rigel", "Taurus", "Aldebaran", "Virgo", "Spica"],
            "Circuits": ["Resistor", "Capacitor", "Switch", "Diode", "Battery", "Current", "Voltage", "Ground"],
            "Ecosystems": ["Grass", "Frog", "Snake", "Hawk", "Sunlight", "Soil", "Mushroom", "Water"],
            "Anatomy": ["Heart", "Lung", "Neuron", "Muscle", "Kidney", "Tendon", "Artery", "Nerve"],
            "Materials": ["Copper", "Glass", "Plastic", "Steel", "Ceramic", "Rubber", "Carbon", "Silicon"],
            "Forces": ["Friction", "Gravity", "Tension", "Lift", "Drag", "Torque", "Impulse", "Pressure"],
            "Waves": ["Amplitude", "Period", "Crest", "Trough", "Frequency", "Phase", "Medium", "Echo"],
            "Earth": ["Core", "Mantle", "Crust", "Fault", "Magma", "Mineral", "Glacier", "Delta"],
        },
    },
}

SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_diagrams_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_diagrams_noise_defaults(scene_id=SCENE, apply_prob=0.30)
