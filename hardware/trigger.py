import logging
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TriggerInterface(ABC):
    @abstractmethod
    def start(self):
        pass
        
    @abstractmethod
    def stop(self):
        pass

class WindowsKeyboardTrigger(TriggerInterface):
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.running = False
        self.root = None
        self.thread = None
        self.is_pressed = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_tk, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.root:
            try:
                self.root.quit()
                self.root.update()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run_tk(self):
        import tkinter as tk
        from app.coordinator import TriggerDownEvent, TriggerUpEvent
        
        self.root = tk.Tk()
        self.root.title("XBud Trigger Window")
        self.root.geometry("300x150")
        
        label = tk.Label(self.root, text="Hold SPACE to Talk", font=("Helvetica", 14))
        label.pack(expand=True)
        
        def on_key_press(event):
            if not self.is_pressed:
                self.is_pressed = True
                label.config(text="Listening...", fg="red")
                self.coordinator.enqueue(TriggerDownEvent())
                
        def on_key_release(event):
            if self.is_pressed:
                self.is_pressed = False
                label.config(text="Hold SPACE to Talk", fg="black")
                self.coordinator.enqueue(TriggerUpEvent())
                
        def on_focus_out(event):
            if self.is_pressed:
                self.is_pressed = False
                label.config(text="Hold SPACE to Talk", fg="black")
                self.coordinator.enqueue(TriggerUpEvent())

        def on_close():
            self.running = False
            self.coordinator.enqueue(TriggerUpEvent()) # ensure release

        self.root.bind('<KeyPress-space>', on_key_press)
        self.root.bind('<KeyRelease-space>', on_key_release)
        self.root.bind('<FocusOut>', on_focus_out)
        self.root.protocol("WM_DELETE_WINDOW", on_close)
        
        # Lift to top
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        
        try:
            while self.running:
                self.root.update_idletasks()
                self.root.update()
        except Exception:
            pass
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

class LinuxGPIOTrigger(TriggerInterface):
    def __init__(self, coordinator, pin=17):
        self.coordinator = coordinator
        self.pin = pin
        self.button = None
        
    def start(self):
        try:
            from gpiozero import Button
            from app.coordinator import TriggerDownEvent, TriggerUpEvent
        except ImportError:
            logger.error("gpiozero not installed. Cannot use LinuxGPIOTrigger.")
            return
            
        self.button = Button(self.pin, bounce_time=0.05)
        
        def pressed():
            self.coordinator.enqueue(TriggerDownEvent())
            
        def released():
            self.coordinator.enqueue(TriggerUpEvent())
            
        self.button.when_pressed = pressed
        self.button.when_released = released
        logger.info(f"Linux GPIO trigger active on pin {self.pin}")

    def stop(self):
        if self.button:
            self.button.close()
            self.button = None
