# -*- coding: utf-8 -*-
"""
config.py

Holds all global constants, directory targets, hierarchical mapping tables,
and statistical thresholds for the hierarchical DORA survey capability analysis.
"""

import os
from typing import Dict
from typing import List

# --- General Run Settings ---
RANDOM_SEED: int = 42
BASE_DIR: str = "dora_analysis_output_pymc"

# --- Compartmentalized Subdirectories ---
MODEL_DIR: str = os.path.join(BASE_DIR, "model")
DIAGNOSTICS_DIR: str = os.path.join(BASE_DIR, "diagnostics")
CSV_DIR: str = os.path.join(BASE_DIR, "estimates_csv")
PLOTS_DIR: str = os.path.join(BASE_DIR, "plots")
LOWRES_PLOTS_DIR: str = os.path.join(BASE_DIR, "lowres_plots")

# Auto-generate folder structure on system load
for d in [MODEL_DIR, DIAGNOSTICS_DIR, CSV_DIR, PLOTS_DIR, LOWRES_PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- MCMC Sampler Configurations ---
ITER_WARMUP: int = 200  # Increased warmup steps as a structural crutch for sparse data
ITER_SAMPLING: int = 150
CHAINS: int = 4  # 8 chains takes too much vram when running posterior checks
TARGET_ACCEPT: float = 0.95  # High target accept to prevent divergent transitions

# --- Regularization Hyperparameters ---
STUDENT_T_NU: float = 5.0  # Degrees of freedom for robust heavy-tailed team baselines
LKJ_ETA: float = 10.0  # Squeezes section correlations toward zero (strong regularization)
SIGMA_CAT_PRIOR_SIGMA: float = 0.25  # Tight Half-Normal prior for category-level offsets
PPC_DRAWS: int = 200  # Thinned draws for posterior predictive checks to prevent VRAM crashes

# --- Statistical Diagnostic Thresholds ---
RHAT_THRESHOLD: float = 1.05
NEFF_RATIO_THRESHOLD: float = 0.1  # Warning flag if ESS/total draws < 10%
PARETO_K_THRESHOLD: float = 0.7  # For LOOIC diagnostic plotting [1]
LOW_DISCRIMINATION_THRESHOLD: float = 0.3
HIGH_CORR_THRESHOLD: float = 0.85  # Threshold for recommending merging of survey dimensions [1]
MIN_RESPONSES_PER_GROUP: int = 5

# --- Demographics & Survey Scales ---
RESPONSE_OPTIONS: List[int] = [1, 2, 3, 4, 5, 6]
N_CATEGORIES_RESPONSE: int = len(RESPONSE_OPTIONS)
YEAR_COL: str = "year"
ID_VAR: str = "team_id"

# --------------------------------------------------------------------
# CHANGE ALL VALUES BELOW IN PROD!
# --------------------------------------------------------------------


# --- Hierarchical Survey Map (Sections -> Categories -> Questions) ---
SURVEY_HIERARCHY: Dict[str, Dict[str, List[str]]] = {

    # -----------------------------------------------------------------------------
    # 'AI and I' Section
    # -----------------------------------------------------------------------------
    'AI and I': {
        'About your effort working with AI': [
            'Considering your responsibility for a final AI-assisted output, including its ethical implications.',
            'Critically evaluating an AI\'s response for accuracy.',
            'Providing an AI with key details like your vision, purpose, or success criteria.',
            'Weighing the pros and cons (e.g., time, quality, effort) when deciding whether to use AI for a task.',
        ],
        'Code Quality': [
            'In the last 3 months, how has AI impacted the quality of your code at work?',
        ],
        'In the last 3 months, how much have you relied on AI for each of the following (15) tasks at work.': [
            'Acquiring / cleaning / analyzing data.',
            'Coding / system design / building and deployment.',
            'Day 2 ops, helpdesk / user support, troubleshooting, report generation (e.g., metrics for staffing).',
            'Reviewing code / reviewing system architecture.',
            'Writing documentation.',
            'Mentoring, interviewing, or helping onboard new employees.',
            'Learning: tech courses / accreditation, tech sharing sessions, conferences (e.g., CPE).',
            'Sprint planning, backlog grooming, standups, brainstorming, strategizing etc.',
            'Attending meetings, engagements, forums, townhalls, cascades.',
            'People management (e.g., writing performance reviews, SAR, 1-on-1 sessions, etc.).',
            'Tech community contribution (e.g., Innohack comm, DevCom, DevForge, etc.).',
            'Clear tech debt.',
            'Administrative tasks (e.g., secretarial, liaison, vendors, budget, events, correspondence, etc.).',
            'Finding and reading relevant literature (e.g., academic, industry research, Gartner).',
            'Product design: user research, elicitation, requirement analysis, project / design discussions, product discovery, user journey mapping, etc.',
        ],
        'Organization Support and AI Adoption': [
            'To what extent does your organization support you with experimenting with AI?',
            'In the last 3 months, my team has been actively adding AI-powered experiences, such as chatbots, for the end users of the primary application or service that I work on.',
        ],

        # # SKIP: boolean question
        # 'Prompt Saving': [
        #     # 'We are interested in learning more about how you treat the prompts you use when working with AI. Thinking about the tasks that have some complexity (for example, refactoring a code base, adding tests, or writing documentation). How do you store the prompts used? (please select all that apply)',
        #     # 'Prompt saving: text',
        #     # 'Prompt saving: not saved',
        #     # 'Prompt saving: code',
        #     # 'Prompt saving: local',
        #     # 'Prompt saving: confluence',
        #     # 'Prompt saving: ephemeral',
        #     # 'Prompt saving: NA',
        # # ],

        'Prompt Style': [
            'After an AI provides an initial output, how frequently do you use follow-up prompts to refine it?',
            'How frequently do you instruct the AI to adopt a specific role or persona?',
            'When giving the AI a complex problem, how frequently do you break it down into smaller, separate prompts yourself?',
        ],

        # # SKIP: open ended
        # 'Your Thoughts about AI and I': [
        #     # '[Optional] What have we done well for AI and I?',
        #     # '[Optional] How do you think we could strengthen AI and I?',
        # # ],
    },

    # -----------------------------------------------------------------------------
    # 'Demographic' Section (Skipped in survey processing)
    # -----------------------------------------------------------------------------
    # # SKIP: demographics
    # 'Demographic': {
    #     'Demographic': [
    #         '(Optional) If you don\'t mind sharing, what do you consider to be your primary application, service, or product?',
    #         # 'My Section',
    #         'My Tech Community',
    #         # 'My Department/Section',
    #         'I am from',
    #         'year', 'dept', 'cluster', 'is_management',
    #     ],
    # # },

    # -----------------------------------------------------------------------------
    # 'Engineering Culture' Section
    # -----------------------------------------------------------------------------
    'Engineering Culture': {
        'Knowledge Sharing': [
            'I rarely find myself needing to answer the same questions that I\'ve already answered before.',
            'I rarely find my work blocked (or otherwise interrupted) because I am waiting on answers to questions.',
        ],
        'User-centricity': [
            'My team has a clear understanding of our users\' goals and is evaluated on our success in delivering value to both our users and our organization.',

            # (2025)
            'My team\'s success is evaluated according to the value we provide to our users and our organization.',
            'Specifications (e.g. requirements, planning) are continuously revisited and reprioritized according to user signals/feedback/telemetry.',
        ],
        'Westrum Organizational Culture': [
            'Cross-functional collaboration is encouraged and rewarded.',
            'Failures are treated primarily as opportunities to improve the system.',
            'Information is actively sought.',
            'Messengers are not punished when they deliver news of failures or other bad news.',
            'New ideas are welcomed.',
            'Responsibilities are shared.',
        ],
        'Work Distribution': [
            'Engineering tasks are distributed evenly/fairly on my team.',
            'My team has a formal process to equitably distribute burdensome tasks / toil / day2 ops / support duties.',
        ],
        'Work Flexibility': [
            'My team provides flexibility in terms of how, when, and where we work.',
        ],

        # # SKIP: open ended
        # 'Your Thoughts about Engineering Culture': [
        #     # '[Optional] What have we done well for Engineering Culture?',
        #     # '[Optional] How do you think we could grow our Engineering Culture?\n',
        # # ],
    },

    # -----------------------------------------------------------------------------
    # 'Key Outcomes' Section
    # -----------------------------------------------------------------------------
    'Key Outcomes': {
        'How often does the primary application or service you work on encounter the following events?': [
            'End users feedback/complain about being dissatisfied with the reliability of our system.',
        ],
        'Organizational Performance': [
            'Achieving our team or mission goals.',
            'Meeting target user adoption (planned users).',
            'Quality of products or services provided.',
        ],
        'Software Delivery Performance': [
            'Deployment Frequency',
            'Change Lead Time',
            'Failed Deployment Recovery Time',
            'Change Failure Rate',
        ],
        'Team Performance': [
            'We delivered innovative/novel solutions.',

            # (2025)
            'We were able to adapt to change.',
            'We were able to effectively collaborate with each other.',
            'We were able to rely on each other.',
            'We worked efficiently.',
        ],
        'We\'re interested about reliability and how both you and your team think about it. For the primary application or service you work on, please rate how strongly you agree or disagree with each of the following statements.': [
            'My team has reliability targets and we regularly review and revise reliability targets based on evidence.',
            'When we miss our reliability targets, we perform improvement work, adjust our development work, and/or re-prioritize.',
            'We regularly test our disaster recovery preparedness through simulated disruptions, failover exercises, table-top exercises, etc.',
            'My team has well-defined procedures for incident management, with blameless post-mortems.',
            'My team works to continuously improve the reliability of an existing system throughout the lifetime of the product, not only during initial design, or immediately after an incident/outage.',
        ],

        # # SKIP: open ended
        # 'Your Thoughts about Key Outcomes': [
        #     # '[Optional] What have we done well for Key Outcomes?',
        #     # '[Optional] How do you think we could better achieve these Key Outcomes?',
        # # ],
    },

    # # SKIP: open ended
    # 'Overall Thoughts': {
    #     'Overall Thoughts of your Engineering Experience': [
    #         # '[Optional] What has improved your engineering experience?',
    #         # '[Optional] What is the biggest problem you faced?',
    #     # ],
    # # },

    # -----------------------------------------------------------------------------
    # 'Processes' Section
    # -----------------------------------------------------------------------------
    'Processes': {
        'Code Complexity': [
            'In the last 6 months, how much did code complexity inside your primary application or service slow down or hinder your development work, if at all?',
        ],
        'Code Review': [
            #'Select the best fitting answer for Code Review',
            'code review is fast',
        ],
        'Continuous Integration / Continuous Delivery': [
            # ci
            'Code commits result in a series of automated tests being run.',
            # (2025) ci
            'Code commits result in an automated build of the software.',
            # cd
            'A failing test will stop the team from deploying the system.',
            # (2025) cd
            'Our software is in a deployable state throughout its lifecycle.',
            'My team prioritizes keeping the software deployable over working on new features.',
        ],
        'Data Ecosystem': [
            'How easily can you discover useful internal data sources you need to complete your work?',
            'How easily can you acquire permission to access internal data sources you need to complete your work?',
            'How easily can you use and analyze the internal data sources you need to complete your work?',
            'How would you rate the overall quality of the data you typically rely on for your work?',
        ],
        'Documentation': [
            'I can rely on our technical documentation when I need to use or work with the services or applications I work on.',
            'It is easy to find the right technical document when I need to understand something about the services or applications I work on.',
            'Technical documentation is updated as changes are made.',
        ],
        'Flexible Infrastructure & Deployments': [
            # flexible infrastructure
            'I can use managed platforms, components, and services to build and run apps / services easily.',
            'I can monitor (and get feedback) and dynamically increase or decrease the resources available for the service or product that I primarily support on demand.',
            # 2025 flexible deployments
            'My team can deploy and release our product or service on demand, independently of other services it depends upon.',
            'On my team, we perform deployments during normal business hours with negligible downtime.',
        ],
        'Loosely Coupled Teams': [
            'How often is your ability to make progress on your work dependent on waiting for input, a review, or a deliverable from another person, team or forum?',
            'My team is able to deliver value to users quickly and independently, without being hindered by dependencies or significant downtime?',
        ],
        'Platform Engineering': [
            'The platform behaves in a way I would expect.',
            'The platform helps me easily build and run my applications and services both reliably and securely.',
            'The platform team acts on the feedback I provide.',
        ],
        'Shift Left on Security': [
            'Is your team aware of common types of vulnerabilities (e.g., OWASP top 10) and are you actively mitigating them?',
            'Is your team performing code scanning (e.g., SAST / static code analysis, secret scanning/detection) and do you actively rectify high priority findings?',
        ],
        'Technical Debt': [
            'In the last 6 months, how much did technical debt inside your primary application or service slow down or hinder your productivity, if at all?',
        ],
        'Version Control': [
            # 'How is version control applied to the following?',
            'How is version control applied to the following? - a. Application code',
            'How is version control applied to the following? - b. Application and system configurations',
            'How is version control applied to the following? - c. Code for automating build and configuration',
            'How is version control applied to the following? - d. Jupyter notebooks',
            'How is version control applied to the following? - e. Scripts for admin, specific day2 tasks, etc',
            'How is version control applied to the following? - f. Prompts for AI systems',
        ],

        # # SKIP: open ended
        # 'Your Thoughts about Process and Technical Capabilities': [
        #     # '[Optional] What have we done well for Processes and Technical Capabilities?',
        #     # '[Optional] How do you think we could strengthen our Processes and Technical Capabilities?',
        # # ],
    },

    # -----------------------------------------------------------------------------
    # 'Well-being' Section
    # -----------------------------------------------------------------------------
    'Well-being': {
        'Engaged': [
            'I feel energized about my work.',

            # (2025)
            'My feelings about work positively influence my life outside of work.',
        ],
        'Flow': [
            'In the last 6 months, how often were you able to reach a high level of focus or reach a state of uninterrupted \'flow\' during development/coding tasks?',
        ],
        'Proficiency': [
            'I am able to do my work in the most effective way possible.',

            # (2025)
            'My work creates value.',
            'My work is aligned with my set of skills.',
        ],
        'Stability': [
            'How stable and how clear are the priorities of your section\'s product line?',
        ],

        # # SKIP: work%
        # 'Work Characteristics': [
        #     # 'How these 15 tasks sum up to 100...',
        #     # 'Work%: Analyzing data',
        #     # 'Work%: Coding / designing / building & deployment',
        #     # 'Work%: Day 2 ops',
        #     # 'Work%: Reviewing code / reviewing system architecture',
        #     # 'Work%: Writing internal/end-user documentation',
        #     # 'Work%: Mentoring',
        #     # 'Work%: Learning: tech courses / accreditation',
        #     # 'Work%: Sprint planning',
        #     # 'Work%: Attending meetings',
        #     # 'Work%: People management',
        #     # 'Work%: Tech community contribution',
        #     # 'Work%: Administrative tasks',
        #     # 'Work%: Clear tech debt',
        #     # 'Work%: Literature review',
        #     # 'Work%: Product design',
        # # ],

        # # SKIP: open ended
        # 'Your Thoughts about Time Allocation and Well-being': [
        #     # '[Optional] What have we done well for Time Allocation and Well-being?',
        #     # '[Optional] How do you think we could improve our Time Allocation and Well-being?',
        # # ],
    },
}

# Categories strictly restricted to engineering personnel (filtered in demographics)
ENGG_ONLY_CATEGORIES: List[str] = [
    "Software Delivery Performance",
    "Continuous Integration / Continuous Delivery",
    "Version Control",
    "Shift Left on Security",
    "Code Review",
    "Flexible Infrastructure & Deployments"
]

# Non-engineering teams to filter out of engineering-only plots
NON_ENGG_TEAMS: List[str] = [
    # --- 1. All Corporate/Corp Lineages ---
    "Corp_Alpha",
    "Corp_Beta",
    "Corp_Gamma",
    "Corp_Delta",
    "Corp_Epsilon",
    "Corp_Zeta",

    # --- 2. Foundational Finance Lineage ---
    "Foundational_Platform_Enterprise_Finance",  # 2025 Parent
    "App_Soln_Enterprise_Finance",               # 2025 & 2026 Descendant

    # --- 3. Foundational Hosting Lineage ---
    "Foundational_Platform_Enterprise_Hosting",  # 2025 Parent
    "App_Soln_Enterprise_Hosting",               # 2025 & 2026 Descendant

    # --- 4. Business Systems Network Lineage ---
    "Biz_Sys_Soln_Net",                          # 2025 & 2026 (No separate parents)

    # --- 5. Supporting/Non-Core Tech Lineages (To even it out) ---
    "Eng_UX_Infra_Hackathon",                    # 2025 & 2026 (Hackathon)
    "Eng_UX_Analytics"                           # 2025 & 2026 (Analytics)
]

# --- Year-by-Year Lineage Transition Map ---
REORG_LINEAGE_MAP = {
    2025: {'App_Cloud_Delivery':            ['Foundational_Cloud_Delivery'],
           'App_Cloud_Prod':                [],
           'App_Cloud_Reliability':         [],
           'App_Soln_Delivery_Soln_Twelve': ['Foundational_Cloud_Delivery'],
           'App_Soln_Enterprise_Delivery':  ['Foundational_Platform_Enterprise_Delivery'],
           'App_Soln_Enterprise_Finance':   ['Foundational_Platform_Enterprise_Finance'],
           'App_Soln_Enterprise_Hosting':   ['Foundational_Platform_Enterprise_Hosting'],
           'Biz_Sys_Soln_Delivery':         ['Data_Web_Ops_Delivery'],
           'Biz_Sys_Soln_Net':              [],
           'Biz_Sys_Soln_Virt':             ['Biz_Sys_Soln_Virt'],
           'Corp_Alpha':                    [],
           'Corp_Beta':                     [],
           'Corp_Gamma':                    [],
           'Data_Soln_Delivery_Prod':       ['Data_Soln_Delivery_Prod'],
           'Data_Sys_API':                  [],
           'Data_Sys_Alpha':                ['Data_Web_Ops_Online'],
           'Data_Sys_Beta':                 ['Data_Web_Ops_Compliance'],
           'Eng_Security_Analytics_Gamma':  [],
           'Eng_Security_Research':         ['Data_Web_Ops_Delivery',
                                             'Data_Web_Ops_Hosting'],
           'Eng_Security_Support':          [],
           'Eng_Soln_Compliance':           [],
           'Eng_Soln_Prod':                 [],
           'Eng_UX_Analytics':              [],
           'Eng_UX_Infra_Eng':              ['Foundational_Soln_Enablement_Support'],
           'Eng_UX_Infra_Hackathon':        [],
           'Eng_UX_Infra_Research':         ['Foundational_Soln_Enablement_Support'],
           'Eng_UX_Prod_B':                 ['Foundational_Soln_Alpha_Telemetry',
                                             'Foundational_Soln_Alpha_Reliability'],
           'Eng_UX_Prod_Compliance':        ['Foundational_Soln_Beta_Prod'],
           'Eng_UX_Prod_Delivery':          [],
           'Eng_UX_Web_Delivery':           ['Foundational_Soln_Beta_Enablement'],
           'Eng_UX_Web_Infra':              ['Foundational_Soln_Alpha_Telemetry',
                                             'Foundational_Soln_Beta_Analytics',
                                             'Foundational_Soln_Beta_Prod'],
           'Eng_UX_Web_Online':             ['Foundational_Soln_Alpha_Telemetry',
                                             'Foundational_Soln_Alpha_Online'],
           },
    2026: {'App_Cloud_Delivery':            ['App_Cloud_Delivery'],
           'App_Cloud_Prod':                ['App_Cloud_Prod'],
           'App_Cloud_Reliability':         ['App_Cloud_Reliability'],
           'App_Soln_Delivery_Soln_Twelve': ['App_Soln_Delivery_Soln_Twelve'],
           'App_Soln_Enterprise_Delivery':  ['App_Soln_Enterprise_Delivery'],
           'App_Soln_Enterprise_Finance':   ['App_Soln_Enterprise_Finance'],
           'App_Soln_Enterprise_Hosting':   ['App_Soln_Enterprise_Hosting'],
           'Biz_Sys_Soln_Delivery':         ['Biz_Sys_Soln_Delivery'],
           'Biz_Sys_Soln_Net':              ['Biz_Sys_Soln_Net'],
           'Biz_Sys_Soln_Virt':             ['Biz_Sys_Soln_Virt'],
           'Corp_Delta':                    ['Corp_Gamma'],
           'Corp_Epsilon':                  ['Corp_Alpha'],
           'Corp_Zeta':                     ['Corp_Beta'],
           'Data_Soln_Delivery_Prod':       ['Data_Soln_Delivery_Prod'],
           'Data_Sys_API':                  ['Data_Sys_API'],
           'Data_Sys_Alpha':                ['Data_Sys_Alpha'],
           'Data_Sys_Beta':                 ['Data_Sys_Beta'],
           'Eng_Security_Analytics_Gamma':  ['Eng_Security_Analytics_Gamma'],
           'Eng_Security_Delivery':         [],
           'Eng_Security_Research':         ['Eng_Security_Research'],
           'Eng_Security_Support':          ['Eng_Security_Support'],
           'Eng_Soln_Compliance':           ['Eng_Soln_Compliance'],
           'Eng_Soln_Prod':                 ['Eng_Soln_Prod'],
           'Eng_UX_Analytics':              ['Eng_UX_Analytics'],
           'Eng_UX_Infra_Eng':              ['Eng_UX_Infra_Eng'],
           'Eng_UX_Infra_Hackathon':        ['Eng_UX_Infra_Hackathon'],
           'Eng_UX_Infra_Research':         ['Eng_UX_Infra_Research'],
           'Eng_UX_Prod_B':                 ['Eng_UX_Prod_B'],
           'Eng_UX_Prod_Compliance':        ['Eng_UX_Prod_Compliance'],
           'Eng_UX_Prod_Delivery':          ['Eng_UX_Prod_Delivery'],
           'Eng_UX_Web_Delivery':           ['Eng_UX_Web_Delivery'],
           'Eng_UX_Web_Infra':              ['Eng_UX_Web_Infra'],
           'Eng_UX_Web_Online':             ['Eng_UX_Web_Online'],
           },
}
