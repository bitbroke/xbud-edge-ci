from abc import ABC, abstractmethod
from typing import Dict, Any

class NotificationSink(ABC):
    @abstractmethod
    def notify(self, reminder: Dict[str, Any]) -> bool:
        """
        Deliver the notification.
        Returns True if delivered successfully, False otherwise.
        """
        pass

class MockNotificationSink(NotificationSink):
    def __init__(self):
        self.deliveries = []
        self.fail_mode = False

    def notify(self, reminder: Dict[str, Any]) -> bool:
        if self.fail_mode:
            return False
        self.deliveries.append(reminder)
        return True
