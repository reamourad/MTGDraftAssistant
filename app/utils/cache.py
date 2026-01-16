"""
Caching utilities for MTG Draft Assistant.

This module provides reusable caching utilities with proper cache
management and cleanup functionality.
"""

import time
import sys
from typing import Dict, Any, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass
from threading import Lock


T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """
    Represents a single cache entry with value and metadata.
    
    Attributes:
        value: The cached value
        timestamp: When the entry was created (seconds since epoch)
        size_bytes: Approximate size of the cached value in bytes
    """
    value: T
    timestamp: float
    size_bytes: int = 0


class CacheManager(Generic[T]):
    """
    Generic cache manager with TTL and size limits.
    
    Features:
    - Time-to-live (TTL) expiration
    - Maximum cache size limits
    - Thread-safe operations
    - Cache statistics
    """
    
    def __init__(
        self,
        ttl_seconds: Optional[int] = None,
        max_size_mb: Optional[int] = None,
        name: str = "cache"
    ):
        """
        Initialize the cache manager.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
            max_size_mb: Maximum cache size in megabytes (None = unlimited)
            name: Name for this cache (for logging/debugging)
        """
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._ttl_seconds = ttl_seconds
        self._max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb else None
        self._name = name
        self._lock = Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                del self._cache[key]
                self._misses += 1
                return None
            
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: T, size_bytes: Optional[int] = None):
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Approximate size in bytes (auto-calculated if None)
        """
        with self._lock:
            # Calculate size if not provided
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            
            # Check if we need to evict entries
            if self._max_size_bytes:
                self._evict_if_needed(size_bytes)
            
            # Store the entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            self._cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """
        Delete a specific cache entry.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        if not self._ttl_seconds:
            return 0
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        size_bytes: Optional[int] = None
    ) -> T:
        """
        Get a value from cache or compute it if not present.
        
        Args:
            key: Cache key
            compute_fn: Function to compute the value if not cached
            size_bytes: Approximate size in bytes (auto-calculated if None)
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache first
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Compute the value
        value = compute_fn()
        
        # Store in cache
        self.set(key, value, size_bytes)
        
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                'name': self._name,
                'entries': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'ttl_seconds': self._ttl_seconds,
                'max_size_mb': self._max_size_bytes / (1024 * 1024) if self._max_size_bytes else None
            }
    
    def _is_expired(self, entry: CacheEntry[T]) -> bool:
        """Check if a cache entry is expired."""
        if not self._ttl_seconds:
            return False
        
        age = time.time() - entry.timestamp
        return age > self._ttl_seconds
    
    def _evict_if_needed(self, new_entry_size: int):
        """
        Evict entries if adding a new entry would exceed size limit.
        
        Uses LRU (Least Recently Used) eviction strategy based on timestamp.
        """
        if not self._max_size_bytes:
            return
        
        current_size = sum(entry.size_bytes for entry in self._cache.values())
        
        # Check if we need to evict
        if current_size + new_entry_size <= self._max_size_bytes:
            return
        
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Evict oldest entries until we have enough space
        for key, entry in sorted_entries:
            if current_size + new_entry_size <= self._max_size_bytes:
                break
            
            del self._cache[key]
            current_size -= entry.size_bytes
            self._evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """
        Estimate the size of a value in bytes.
        
        This is a rough estimate using sys.getsizeof.
        """
        try:
            return sys.getsizeof(value)
        except:
            return 0


class SimpleDictCache(Generic[T]):
    """
    Simple dictionary-based cache without TTL or size limits.
    
    Useful for cases where you just need basic caching without
    the overhead of expiration checking.
    """
    
    def __init__(self, name: str = "simple_cache"):
        """
        Initialize the simple cache.
        
        Args:
            name: Name for this cache (for logging/debugging)
        """
        self._cache: Dict[str, T] = {}
        self._name = name
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache."""
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: T):
        """Set a value in the cache."""
        with self._lock:
            self._cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete a specific cache entry."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Get a value from cache or compute it if not present."""
        cached_value = self.get(key)
        if cached_value is not None:
            return cached_value
        
        value = compute_fn()
        self.set(key, value)
        return value
    
    def size(self) -> int:
        """Get the number of entries in the cache."""
        with self._lock:
            return len(self._cache)


# Global cache manager instance
_cache_manager_instance: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        The global CacheManager instance
    """
    global _cache_manager_instance
    if _cache_manager_instance is None:
        from app.utils.config import get_config
        config = get_config()
        _cache_manager_instance = CacheManager(
            ttl_seconds=config.cache_ttl_seconds,
            max_size_mb=config.max_cache_size_mb,
            name="global"
        )
    return _cache_manager_instance


def set_cache_manager(cache_manager: CacheManager):
    """
    Set the global cache manager instance.
    
    Args:
        cache_manager: New cache manager instance
    """
    global _cache_manager_instance
    _cache_manager_instance = cache_manager
