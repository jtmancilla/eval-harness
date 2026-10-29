"""Vendor-neutral contracts for MAYA synthetic evaluation matrices."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MayaRoute(StrEnum):
    RAG = "RAG"
    IBM_ASSISTANT = "IBM_ASSISTANT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class MayaDisposition(StrEnum):
    ANSWERED = "ANSWERED"
    COMPLETED = "COMPLETED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


class MayaComplexity(StrEnum):
    SIMPLE = "SIMPLE"
    MULTI_ACTION = "MULTI_ACTION"
    ADVERSARIAL = "ADVERSARIAL"


class MayaScenarioKind(StrEnum):
    OPERATION = "OPERATION"
    INFORMATION = "INFORMATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CLARIFICATION = "CLARIFICATION"
    MULTI_ACTION = "MULTI_ACTION"
    ADVERSARIAL = "ADVERSARIAL"


class MayaSegment(StrEnum):
    PERSONAL = "PERSONAL"
    BUSINESS = "BUSINESS"


class MayaChannel(StrEnum):
    PUBLIC_WEB = "PUBLIC_WEB"
    BANK_WEB = "BANK_WEB"
    MOBILE = "MOBILE"
    BEM = "BEM"
    BEP = "BEP"
    BEM_MOBILE = "BEM_MOBILE"


class MayaUrgency(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DigitalLiteracy(StrEnum):
    BASIC = "BASIC"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"


class MayaWriteTarget(StrEnum):
    DATABASE = "DATABASE"
    AUDIT_LOG = "AUDIT_LOG"
    OPERATION_LOG = "OPERATION_LOG"
    SECURITY_LOG = "SECURITY_LOG"
    NOTIFICATION = "NOTIFICATION"


class MayaVerdict(StrEnum):
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"


class MayaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MayaPersona(MayaModel):
    persona_id: str = Field(pattern=r"^p_[a-z0-9_]+$")
    region: str
    state: str
    city: str
    age: int = Field(ge=18, le=100)
    segment: MayaSegment
    channel: MayaChannel
    digital_literacy: DigitalLiteracy
    urgency: MayaUrgency
    communication_preference: str


class MayaNarrativeStyle(MayaModel):
    style_id: str = Field(pattern=r"^s_[a-z0-9_]+$")
    description: str
    language_register: str
    noise_level: str = Field(pattern=r"^(low|medium|high)$")


class MayaToolParameter(MayaModel):
    name: str
    value_type: str
    description: str
    allowed_values: tuple[str, ...] = ()


class MayaToolSpec(MayaModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    operation_id: int | None = Field(default=None, ge=1, le=17)
    description: str
    parameters: tuple[MayaToolParameter, ...] = ()
    required: tuple[str, ...] = ()
    canonical: bool
    target_canonical_tool: str | None = None

    @model_validator(mode="after")
    def validate_schema(self) -> Self:
        parameter_names = {parameter.name for parameter in self.parameters}
        if len(parameter_names) != len(self.parameters):
            raise ValueError("Tool parameter names must be unique")
        if not set(self.required).issubset(parameter_names):
            raise ValueError("Required parameters must be declared")
        if not self.canonical and self.target_canonical_tool is None:
            raise ValueError("Decoy tools must declare a canonical target")
        return self


class MayaKnowledgeSnippet(MayaModel):
    document_id: str
    title: str
    content: str


class MayaSyntheticContext(MayaModel):
    customer_id: str = Field(pattern=r"^SYN-CUSTOMER-[A-Z0-9_]+$")
    product_ids: tuple[str, ...]
    authenticated: bool
    channel_profile_complete: bool
    token_active: bool
    registered_device: bool
    sufficient_funds: bool
    eligible_purchase: bool


class MayaCase(MayaModel):
    case_id: str
    schema_version: str
    scenario_id: str
    kind: MayaScenarioKind
    complexity: MayaComplexity
    persona: MayaPersona
    style: MayaNarrativeStyle
    user_text: str = Field(min_length=1)
    context: MayaSyntheticContext
    available_tools: tuple[MayaToolSpec, ...] = Field(min_length=1)
    knowledge_snippets: tuple[MayaKnowledgeSnippet, ...] = ()
    saturation_level: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_tool_catalog(self) -> Self:
        tool_names = [tool.name for tool in self.available_tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("Available tool names must be unique")
        if len(tool_names) != self.saturation_level:
            raise ValueError("Saturation level must match the available tool count")
        return self


class MayaActionArgument(MayaModel):
    name: str
    value: str | int | float | bool


class MayaExpectedAction(MayaModel):
    action_id: str
    operation_id: int | None = Field(default=None, ge=1, le=17)
    tool_name: str
    arguments: tuple[MayaActionArgument, ...] = ()
    depends_on: tuple[str, ...] = ()
    should_execute: bool = True

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("Action dependencies must be unique")
        if self.action_id in self.depends_on:
            raise ValueError("An action cannot depend on itself")
        argument_names = [argument.name for argument in self.arguments]
        if len(argument_names) != len(set(argument_names)):
            raise ValueError("Action argument names must be unique")
        return self


class MayaValidationExpectation(MayaModel):
    validation_id: str
    action_id: str
    description: str
    passes: bool


class MayaWriteExpectation(MayaModel):
    write_id: str
    action_id: str
    target: MayaWriteTarget
    record_type: str
    description: str


class MayaResponseExpectation(MayaModel):
    language: str = "es-MX"
    required_facts: tuple[str, ...] = ()
    forbidden_content: tuple[str, ...] = ()
    must_request_clarification: bool = False
    tone: tuple[str, ...] = ("claro", "respetuoso")


class MayaObservedAction(MayaModel):
    action_id: str
    tool_name: str = ""
    arguments: tuple[MayaActionArgument, ...] = ()
    executed: bool = False


class MayaObservedValidation(MayaModel):
    validation_id: str
    passed: bool | None = None


class MayaObservedWrite(MayaModel):
    write_id: str
    target: MayaWriteTarget | None = None
    record_type: str = ""
    occurred: bool = False


class MayaObservation(MayaModel):
    case_id: str
    response_text: str = ""
    route: MayaRoute | None = None
    actions: tuple[MayaObservedAction, ...] = ()
    validations: tuple[MayaObservedValidation, ...] = ()
    writes: tuple[MayaObservedWrite, ...] = ()
    reviewer_notes: str = ""


class MayaScorecard(MayaModel):
    case_id: str
    route_correct: bool | None = None
    action_selection_correct: bool | None = None
    arguments_correct: bool | None = None
    execution_order_correct: bool | None = None
    validations_correct: bool | None = None
    persistence_and_logs_correct: bool | None = None
    response_correct: bool | None = None
    final_verdict: MayaVerdict = MayaVerdict.PENDING
    reviewer_notes: str = ""


def validate_action_graph(
    actions: tuple[MayaExpectedAction, ...],
    validations: tuple[MayaValidationExpectation, ...],
    writes: tuple[MayaWriteExpectation, ...],
    forbidden_writes: tuple[MayaWriteExpectation, ...],
) -> None:
    actions_by_id = {action.action_id: action for action in actions}
    if len(actions_by_id) != len(actions):
        raise ValueError("Action IDs must be unique")

    for action in actions:
        missing = set(action.depends_on) - set(actions_by_id)
        if missing:
            raise ValueError(f"Unknown action dependencies: {sorted(missing)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(action_id: str) -> None:
        if action_id in visiting:
            raise ValueError("Action dependencies must be acyclic")
        if action_id in visited:
            return
        visiting.add(action_id)
        for dependency in actions_by_id[action_id].depends_on:
            visit(dependency)
        visiting.remove(action_id)
        visited.add(action_id)

    for action_id in actions_by_id:
        visit(action_id)

    positions = {action.action_id: index for index, action in enumerate(actions)}
    for action in actions:
        for dependency in action.depends_on:
            if positions[dependency] >= positions[action.action_id]:
                raise ValueError("Actions must be listed in topological execution order")

    for validation in validations:
        if validation.action_id not in actions_by_id:
            raise ValueError(f"Unknown action reference: {validation.action_id}")
    for write in (*writes, *forbidden_writes):
        if write.action_id not in actions_by_id:
            raise ValueError(f"Unknown action reference: {write.action_id}")

    validation_ids = [validation.validation_id for validation in validations]
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Validation IDs must be unique")
    write_ids = [write.write_id for write in writes]
    if len(write_ids) != len(set(write_ids)):
        raise ValueError("Write IDs must be unique")
    forbidden_write_ids = [write.write_id for write in forbidden_writes]
    if len(forbidden_write_ids) != len(set(forbidden_write_ids)):
        raise ValueError("Forbidden write IDs must be unique")

    failed_actions = {validation.action_id for validation in validations if not validation.passes}
    blocked_actions = set(failed_actions)
    for action in actions:
        if any(dependency in blocked_actions for dependency in action.depends_on):
            blocked_actions.add(action.action_id)
        if action.action_id in blocked_actions and action.should_execute:
            raise ValueError("Failed validations must block the action and its dependants")


class MayaOutcomeRules(MayaModel):
    actions: tuple[MayaExpectedAction, ...] = ()
    validations: tuple[MayaValidationExpectation, ...] = ()
    writes: tuple[MayaWriteExpectation, ...] = ()
    forbidden_writes: tuple[MayaWriteExpectation, ...] = ()
    response: MayaResponseExpectation

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        validate_action_graph(
            self.actions, self.validations, self.writes, self.forbidden_writes
        )
        return self


class MayaScenarioTemplate(MayaOutcomeRules):
    scenario_id: str
    title: str
    kind: MayaScenarioKind
    complexity: MayaComplexity
    utterances: tuple[str, ...] = Field(min_length=1)
    expected_route: MayaRoute
    disposition: MayaDisposition
    knowledge_snippets: tuple[MayaKnowledgeSnippet, ...] = ()
    authenticated: bool = True
    channel_profile_complete: bool = True
    token_active: bool = True
    registered_device: bool = True
    sufficient_funds: bool = True
    eligible_purchase: bool = True


class MayaExpectedOutcome(MayaOutcomeRules):
    case_id: str
    expected_route: MayaRoute
    disposition: MayaDisposition
