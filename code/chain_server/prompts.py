# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The collection of prompts used in this application."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

# Condense Question Prompt Template
CONDENSE_QUESTION_TEMPLATE = PromptTemplate.from_template(
    """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is. Don't frame your response.
    Chat History:
    {history}
    Follow Up question: {question}
    Standalone question:"""
)

# Health Search Bot Prompt
HEALTH_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an AI assistant designed to help with health-related literature searches. "
            "Using the PICO framework, assist in extracting key components from the text, "
            "and formulate a suitable search query for databases like PubMed. You are using a "
            "Retrieval-Augmented Generation (RAG) model with NVIDIA NIMs to identify relevant information."
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "{question}"
        ),
        (
            "assistant",
            "Based on the provided information, I will guide you through the PICO process:
            - **Patient/Population/Problem (P)**: Identify who the patient or population is, and what their health problem is.
            - **Intervention (I)**: Determine the intervention of interest (e.g., treatment, exposure).
            - **Comparison (C)**: Specify any comparison interventions (if applicable).
            - **Outcome (O)**: Define the outcomes of interest (e.g., reduction in symptoms, prevention).
            Please provide the necessary details or confirm that you would like me to proceed with formulating the PubMed query."
        ),
    ]
)

PROMPTS = {
    "condense": CONDENSE_QUESTION_TEMPLATE,
    "health_search": HEALTH_SEARCH_PROMPT,
}
