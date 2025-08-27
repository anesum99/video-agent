#!/usr/bin/env python3
"""
Minimal test to isolate the frames attribute error
"""

import sys
import traceback
from typing import Dict, Any

# Mock the dependencies we don't have
class MockImage:
    def __init__(self):
        self.width = 100
        self.height = 100

class MockProcessor:
    def __init__(self):
        pass

class MockModel:
    def __init__(self):
        pass

# Mock the imports
sys.modules['torch'] = type('MockTorch', (), {
    'cuda': type('MockCuda', (), {'is_available': lambda *args: False})(),
    'float32': 'float32',
    'float16': 'float16',
    'no_grad': lambda *args: type('MockContext', (), {'__enter__': lambda self: None, '__exit__': lambda *args: None})()
})()

sys.modules['av'] = type('MockAv', (), {
    'open': lambda path: type('MockContainer', (), {
        'duration': 10.0,
        'streams': type('MockStreams', (), {
            'video': [type('MockStream', (), {
                'frames': 300,
                'width': 640,
                'height': 480,
                'codec_context': type('MockCodec', (), {'name': 'h264'})(),
                'bit_rate': 1000000,
                'average_rate': 30.0
            })()]
        })(),
        'close': lambda: None
    })(),
    'time_base': 1/90000
})()

sys.modules['transformers'] = type('MockTransformers', (), {
    'VideoLlavaProcessor': type('MockProcessorClass', (), {
        'from_pretrained': lambda model: MockProcessor()
    }),
    'VideoLlavaForConditionalGeneration': type('MockModelClass', (), {
        'from_pretrained': lambda model, **kwargs: MockModel()
    }),
    'AutoTokenizer': type('MockTokenizer', (), {
        'from_pretrained': lambda model, **kwargs: type('MockTok', (), {'pad_token': None, 'eos_token': '<eos>'})()
    }),
    'AutoModelForCausalLM': type('MockAutoModel', (), {
        'from_pretrained': lambda model, **kwargs: MockModel()
    })
})()

sys.modules['numpy'] = type('MockNumPy', (), {
    'linspace': lambda start, stop, num, dtype=None: list(range(num)),
    'array': lambda x: x
})()

sys.modules['gradio'] = type('MockGradio', (), {})()
sys.modules['langgraph.graph'] = type('MockLanggraph', (), {
    'StateGraph': type('MockStateGraph', (), {
        '__init__': lambda self, state_cls: None,
        'add_node': lambda self, name, func: None,
        'add_edge': lambda self, from_node, to_node: None,
        'add_conditional_edges': lambda self, from_node, condition, mapping: None,
        'compile': lambda self, **kwargs: type('MockGraph', (), {
            'invoke': lambda self, state, config: state
        })()
    }),
    'START': 'START',
    'END': 'END'
})()

sys.modules['langgraph.checkpoint.memory'] = type('MockMemory', (), {
    'MemorySaver': lambda: None
})()

sys.modules['langchain_core.messages'] = type('MockMessages', (), {
    'HumanMessage': type('MockHumanMessage', (), {}),
    'AIMessage': type('MockAIMessage', (), {}),
    'BaseMessage': type('MockBaseMessage', (), {})
})()

# Now try to import and test our agent
try:
    from video_agent_mcp_full import AgentState, VideoAgent, ModelConfig, AgentConfig
    
    print("✅ Import successful")
    
    # Test AgentState creation
    state = AgentState(
        video_path="test.mp4",
        question="How many people?",
        frames_b64=["fake_base64_data"]
    )
    
    print("✅ AgentState creation successful")
    print(f"State has frames_b64: {hasattr(state, 'frames_b64')}")
    print(f"State has frames: {hasattr(state, 'frames')}")
    
    # Test get_frames method
    try:
        frames = state.get_frames()
        print("✅ get_frames() works")
    except Exception as e:
        print(f"❌ get_frames() failed: {e}")
    
    # Test agent creation
    try:
        config = ModelConfig(max_frames=8, max_new_tokens=50)
        agent_config = AgentConfig()
        agent = VideoAgent(config, agent_config)
        print("✅ VideoAgent creation successful")
        
        # Test processing (this is where the error likely occurs)
        result = agent.process("test.mp4", "How many people?")
        print(f"✅ Agent processing successful: {result}")
        
    except Exception as e:
        print(f"❌ Agent creation/processing failed: {e}")
        traceback.print_exc()

except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()