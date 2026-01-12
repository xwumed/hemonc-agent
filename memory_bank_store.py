from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os
from datetime import datetime

@dataclass
class MemoryEntry:
    """Represents a single interaction in the memory bank"""
    timestamp: str
    patient_info: str
    final_decision: str
    metadata: Dict[str, Any] = None


class MemoryBank:
    """Improved memory bank that maintains conversation context without complex prompts"""

    def __init__(self, patient_id: str, storage_dir: str = "memory_bank"):
        self.patient_id = patient_id
        # ensure the folder exists
        os.makedirs(storage_dir, exist_ok=True)
        # now memory_file lives in that folder
        self.memory_file = os.path.join(storage_dir, f"memory_bank_{patient_id}.json")
        self.entries: List[MemoryEntry] = []
        self.max_entries = int(os.getenv("MEMORY_MAX_ENTRIES", "3"))
        self.load_memory()

    def load_memory(self):
        """Load existing memory if available"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.entries = [
                        MemoryEntry(**entry) for entry in data.get('entries', [])
                    ]
            except Exception as e:
                print(f"Warning: Could not load memory bank: {e}")
                self.entries = []

    def save_memory(self):
        """Save current memory state"""
        data = {
            'patient_id': self.patient_id,
            'last_updated': datetime.now().isoformat(),
            'entries': [
                {
                    'timestamp': entry.timestamp,
                    'patient_info': entry.patient_info,
                    'final_decision': entry.final_decision,
                    'metadata': entry.metadata or {}
                }
                for entry in self.entries
            ]
        }
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: failed to save memory bank {self.memory_file}: {e}")

    def is_first_interaction(self) -> bool:
        """Check if this is the first interaction"""
        return len(self.entries) == 0

    def get_context_for_agents(self, current_info: str) -> str:
        """Build context string for agents without exposing internal structure"""
        if self.is_first_interaction():
            return current_info

        # For follow-up interactions, include all previous interactions
        context = "=== HISTORY INFORMATION ===\n\n"

        history = self.entries[-self.max_entries :]
        for i, entry in enumerate(history, 1):
            context += f"Assessment {i} ({entry.timestamp}):\n"
            context += f"Patient Info:\n{entry.patient_info}\n\n"
            # context += f"Final Decision: {entry.final_decision}\n\n"

        context += "=== CURRENT INFORMATION ===\n"
        context += current_info
        return context

    def get_context_for_prechecks(self, current_info: str) -> str:
        """Build context for precheck agents, including only patient info from all previous interactions"""
        if self.is_first_interaction():
            return current_info

        # For follow-up interactions, include all previous patient info without final decisions
        context = "=== HISTORY INFORMATION ===\n\n"

        history = self.entries[-self.max_entries :]
        for i, entry in enumerate(history, 1):
            context += f"Assessment {i} ({entry.timestamp}):\n"
            context += f"{entry.patient_info}\n\n"

        context += "=== CURRENT INFORMATION ===\n"
        context += current_info
        return context

    def add_entry(self, timestamp: str, patient_info: str,
                  final_decision: str, metadata: Dict[str, Any] = None):
        """Add a new memory entry"""
        entry = MemoryEntry(
            timestamp=timestamp,
            patient_info=patient_info,
            final_decision=final_decision,
            metadata=metadata
        )
        self.entries.append(entry)
        self.save_memory()

    def get_summary(self) -> str:
        """Get a summary of all interactions"""
        if not self.entries:
            return "No previous interactions recorded."

        summary = f"Patient ID: {self.patient_id}\n"
        summary += f"Total interactions: {len(self.entries)}\n\n"

        for i, entry in enumerate(self.entries, 1):
            summary += f"Interaction {i} ({entry.timestamp}):\n"
            summary += f"Decision: {entry.final_decision[:100]}...\n\n"

        return summary