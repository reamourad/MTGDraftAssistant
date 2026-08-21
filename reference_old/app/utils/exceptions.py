"""
Centralized error handling for MTG Draft Assistant.

This module defines domain-specific exception classes and implements
consistent error handling patterns across the application.
"""

from typing import Optional, Dict, Any


class MTGDraftError(Exception):
    """
    Base exception for all MTG Draft Assistant errors.
    
    All custom exceptions should inherit from this class to allow
    for consistent error handling and logging.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Additional context about the error
            original_error: Original exception if this is wrapping another error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dictionary with error information
        """
        result = {
            'error': self.__class__.__name__,
            'message': self.message
        }
        
        if self.details:
            result['details'] = self.details
        
        if self.original_error:
            result['original_error'] = str(self.original_error)
        
        return result


class ModelNotFoundError(MTGDraftError):
    """
    Raised when a trained model for a set is not found.
    
    This typically occurs when trying to make predictions for a set
    that hasn't been trained yet.
    """
    
    def __init__(
        self,
        set_code: str,
        model_path: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            set_code: The set code that was requested
            model_path: Path where the model was expected
            original_error: Original exception if applicable
        """
        message = f"No trained model found for set '{set_code}'"
        if model_path:
            message += f" at path '{model_path}'"
        
        details = {'set_code': set_code}
        if model_path:
            details['model_path'] = model_path
        
        super().__init__(message, details, original_error)
        self.set_code = set_code
        self.model_path = model_path


class InvalidSetError(MTGDraftError):
    """
    Raised when an invalid or unsupported set code is provided.
    
    This occurs when a set code doesn't exist or isn't supported
    by the application.
    """
    
    def __init__(
        self,
        set_code: str,
        available_sets: Optional[list] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            set_code: The invalid set code
            available_sets: List of valid set codes (optional)
            original_error: Original exception if applicable
        """
        message = f"Invalid or unsupported set code: '{set_code}'"
        if available_sets:
            message += f". Available sets: {', '.join(available_sets)}"
        
        details = {'set_code': set_code}
        if available_sets:
            details['available_sets'] = available_sets
        
        super().__init__(message, details, original_error)
        self.set_code = set_code
        self.available_sets = available_sets


class PredictionError(MTGDraftError):
    """
    Raised when prediction fails for any reason.
    
    This is a general error for prediction-related failures,
    including model inference errors, invalid input, etc.
    """
    
    def __init__(
        self,
        message: str,
        set_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            set_code: Set code being predicted (optional)
            details: Additional context
            original_error: Original exception if applicable
        """
        full_details = details or {}
        if set_code:
            full_details['set_code'] = set_code
        
        super().__init__(message, full_details, original_error)
        self.set_code = set_code


class DataNotFoundError(MTGDraftError):
    """
    Raised when required data files are not found.
    
    This includes missing card data, booster configs, sheets, etc.
    """
    
    def __init__(
        self,
        data_type: str,
        set_code: Optional[str] = None,
        file_path: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            data_type: Type of data that wasn't found (e.g., 'cards', 'booster_config')
            set_code: Set code (optional)
            file_path: Expected file path (optional)
            original_error: Original exception if applicable
        """
        message = f"Required data not found: {data_type}"
        if set_code:
            message += f" for set '{set_code}'"
        if file_path:
            message += f" at path '{file_path}'"
        
        details = {'data_type': data_type}
        if set_code:
            details['set_code'] = set_code
        if file_path:
            details['file_path'] = file_path
        
        super().__init__(message, details, original_error)
        self.data_type = data_type
        self.set_code = set_code
        self.file_path = file_path


class CardNotFoundError(MTGDraftError):
    """
    Raised when a specific card is not found in a set.
    
    This occurs when trying to look up a card that doesn't exist
    in the specified set.
    """
    
    def __init__(
        self,
        card_name: str,
        set_code: str,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            card_name: Name of the card that wasn't found
            set_code: Set code where the card was expected
            original_error: Original exception if applicable
        """
        message = f"Card '{card_name}' not found in set '{set_code}'"
        details = {
            'card_name': card_name,
            'set_code': set_code
        }
        
        super().__init__(message, details, original_error)
        self.card_name = card_name
        self.set_code = set_code


class EncodingError(MTGDraftError):
    """
    Raised when card encoding fails.
    
    This occurs when the card encoder cannot properly encode
    a card's features.
    """
    
    def __init__(
        self,
        card_name: Optional[str] = None,
        message: str = "Card encoding failed",
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            card_name: Name of the card that failed to encode (optional)
            message: Error message
            details: Additional context
            original_error: Original exception if applicable
        """
        full_details = details or {}
        if card_name:
            full_details['card_name'] = card_name
        
        super().__init__(message, full_details, original_error)
        self.card_name = card_name


class CacheError(MTGDraftError):
    """
    Raised when cache operations fail.
    
    This includes cache read/write failures, eviction errors, etc.
    """
    
    def __init__(
        self,
        operation: str,
        cache_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            operation: Cache operation that failed (e.g., 'get', 'set', 'evict')
            cache_key: Cache key involved (optional)
            details: Additional context
            original_error: Original exception if applicable
        """
        message = f"Cache operation '{operation}' failed"
        if cache_key:
            message += f" for key '{cache_key}'"
        
        full_details = details or {}
        full_details['operation'] = operation
        if cache_key:
            full_details['cache_key'] = cache_key
        
        super().__init__(message, full_details, original_error)
        self.operation = operation
        self.cache_key = cache_key


class ValidationError(MTGDraftError):
    """
    Raised when input validation fails.
    
    This includes invalid request parameters, malformed data, etc.
    """
    
    def __init__(
        self,
        field: str,
        message: str,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            field: Field that failed validation
            message: Validation error message
            value: Invalid value (optional)
            details: Additional context
            original_error: Original exception if applicable
        """
        full_message = f"Validation failed for field '{field}': {message}"
        
        full_details = details or {}
        full_details['field'] = field
        if value is not None:
            full_details['value'] = value
        
        super().__init__(full_message, full_details, original_error)
        self.field = field
        self.value = value


class ConfigurationError(MTGDraftError):
    """
    Raised when there's a configuration error.
    
    This includes missing configuration values, invalid settings, etc.
    """
    
    def __init__(
        self,
        config_key: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            config_key: Configuration key that's problematic
            message: Error message
            details: Additional context
            original_error: Original exception if applicable
        """
        full_message = f"Configuration error for '{config_key}': {message}"
        
        full_details = details or {}
        full_details['config_key'] = config_key
        
        super().__init__(full_message, full_details, original_error)
        self.config_key = config_key


class BoosterGenerationError(MTGDraftError):
    """
    Raised when booster pack generation fails.
    
    This includes missing booster configs, invalid sheets, etc.
    """
    
    def __init__(
        self,
        set_code: str,
        message: str = "Booster generation failed",
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            set_code: Set code for which booster generation failed
            message: Error message
            details: Additional context
            original_error: Original exception if applicable
        """
        full_details = details or {}
        full_details['set_code'] = set_code
        
        super().__init__(message, full_details, original_error)
        self.set_code = set_code
