"""
Services Layer - External integrations and service interfaces.

This module contains services for external API integrations
and other external service dependencies.
"""

from app.services.mtgjson import MTGJsonService

__all__ = ['MTGJsonService']