import chainlit.data as cl_data
from typing import Dict, List, Optional, Any
import json
from chainlit.types import ThreadDict, Feedback, PageInfo
from chainlit.user import UserDict, PersistedUser
from chainlit.element import ElementDict
from chainlit.step import StepDict
from chainlit.data.base import Pagination, ThreadFilter, PaginatedResponse
from chainlit.data.utils import queue_until_user_message
from pathlib import Path
from datetime import datetime, timezone
from config import config

# 打印调试信息的开关
DEBUG_MODE = False

# 添加获取UTC时间的函数
def utc_now() -> str:
    """获取当前UTC时间并格式化为ISO格式字符串"""
    return datetime.now(timezone.utc).isoformat()

def debug_log(message: str) -> None:
    """调试日志打印函数"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

class AI4FSDataLayer(cl_data.BaseDataLayer):
    def __init__(self):
        debug_log("初始化 AI4FSDataLayer")
        # 设置数据存储文件路径
        self.data_file = Path(config.USER_SESSIONS_FILE)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self._init_data_file()
    
    def _init_data_file(self):
        debug_log("初始化数据文件结构")
        # 初始化数据文件结构
        initial_data = {
            "users": {},
            "threads": {},
            "delete_threads": []
        }
        self._save_data(initial_data)
    
    def _load_data(self):
        debug_log(f"加载数据文件: {self.data_file}")
        # 读取数据文件
        if not self.data_file.exists():
            debug_log("数据文件不存在,创建新文件")
            return self._init_data_file()
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                debug_log("数据文件加载成功")
                return data
        except:
            debug_log("数据文件加载失败,重新初始化")
            return self._init_data_file()
    
    def _save_data(self, data):
        # 保存数据到文件
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        debug_log(f"获取用户信息: {identifier}")
        data = self._load_data()
        user = data["users"].get(identifier)
        if user:
            debug_log("用户存在,返回用户信息")
            return PersistedUser(
                id=identifier,
                identifier=identifier,
                metadata=user.get("metadata", {}),
                createdAt=user.get("createdAt", utc_now())
            )
        debug_log("用户不存在")
        return None

    async def create_user(self, user: UserDict) -> Optional[PersistedUser]:
        data = self._load_data()
        current_time = utc_now()
        
        persisted_user = PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            metadata=user.metadata,
            createdAt=current_time
        )
        
        data["users"][user.identifier] = {
            "id": user.identifier,
            "identifier": user.identifier,
            "metadata": user.metadata,
            "createdAt": current_time
        }
        self._save_data(data)
        return persisted_user

    async def delete_thread(self, thread_id: str) -> None:
        data = self._load_data()
        thread = data["threads"].pop(thread_id, None)
        debug_log(f"test delete_thread {thread}")
        data["delete_threads"].append(thread_id)
        self._save_data(data)
        
    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        data = self._load_data()
        thread = data["threads"].get(thread_id) or None
        if not thread:
            return None
        
        steps = thread.get("steps", [])
        thread['steps'] = sorted(steps, key=lambda x: x['createdAt'])
        return thread
        
    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        debug_log(f"更新对话: {thread_id}")
        data = self._load_data()
        
        # 检查是否在已删除列表中
        if thread_id in data["delete_threads"]:
            debug_log(f"对话 {thread_id} 已被删除")
            return None
            
        # 注意：userIdentity必须是登录账户，且必须设置，否则不能从历史聊天记录中恢复继续聊天
        admin_user = list(data["users"].values())[0]
        thread = data["threads"].setdefault(thread_id, {
            "id": thread_id,
            "createdAt": utc_now(),
            "userIdentifier": admin_user.get("identifier")
        })
        
        updates = {
            "name": name,
            "userId": user_id,
            "metadata": metadata,
            "tags": tags,
        }
        
        # 只更新非None的值
        thread.update({k: v for k, v in updates.items() if v is not None})
        
        self._save_data(data)
        debug_log("对话更新完成")

    @cl_data.queue_until_user_message()
    async def create_step(self, step: StepDict) -> None:
        datas = self._load_data()
        thread = datas["threads"].get(step["threadId"]) or None
        
        if not thread:
            debug_log(f"create_step: thread {step['threadId']} 不存在")
            return None
        
        thread.setdefault("steps", []).append(step)
        self._save_data(datas)
    
    @queue_until_user_message()
    async def delete_step(self, step_id: str) -> None:
        pass
    
    @queue_until_user_message()
    async def create_element(self, element: ElementDict) -> None:
        pass

    @queue_until_user_message()
    async def delete_element(self, element_id: str) -> None:
        pass

    async def get_element(self, element_id: str) -> Optional[ElementDict]:
        pass
        
    async def get_thread_author(self, thread_id: str) -> Optional[str]:
        data = self._load_data()
        thread = data["threads"].get(thread_id) or {}
        return thread.get("userIdentifier") if thread else None
        
    async def list_threads(
        self,
        pagination: Pagination,
        thread_filter: Optional[ThreadFilter] = None
    ) -> PaginatedResponse[ThreadDict]:
        data = self._load_data()
        
        threads = data.get("threads") or None
        if not threads:
            return PaginatedResponse(
                data=[],
                pageInfo=PageInfo(
                    hasNextPage=False,
                    startCursor=None,
                    endCursor=None
                )
            )

        threads_list = list(threads.values())
        # 获取分页参数
        page_size = pagination.first
        current_cursor = pagination.cursor 
        # 如果有游标，找到开始位置
        start = 0
        if current_cursor:
            try:
                start = int(current_cursor) + 1  # 从游标的下一个位置开始
            except ValueError:
                start = 0
        # 计算结束位置
        end = start + page_size
        # 获取当前页的数据
        paginated_threads = threads_list[start:end]
        # 判断是否还有下一页
        has_next = len(threads_list) > end
        # 设置游标信息
        start_cursor = str(start) if paginated_threads else None
        end_cursor = str(end - 1) if paginated_threads else None
        
        return PaginatedResponse(
            data=paginated_threads,
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=start_cursor,
                endCursor=end_cursor
            )
        )
    
    @queue_until_user_message()
    async def update_step(self, step: StepDict) -> None:
        step_id = step.get("id")
        thread_id = step.get("threadId")
        
        data = self._load_data()
        # 使用链式获取,简化多层判断
        thread = data.get("threads", {}).get(thread_id)
        if not thread:
            debug_log("test update_step: thread不存在")
            return None
        
        steps = thread.get("steps", [])
        if not steps:
            debug_log("test update_step: steps不存在")
            return None
            
        target_step = next((step for step in steps if step["id"] == step_id), None)
        if not target_step:
            return None

        # 使用step更新target_step，重点是要更新input或output
        target_step.update({
            "input": step.get("input"),
            "output": step.get("output"),
            "metadata": step.get("metadata"),
            "feedback": step.get("feedback"),
            "start_time": step.get("start_time"),
            "end_time": step.get("end_time"),
            "error": step.get("error")
        })
        self._save_data(data)
        return None

    async def upsert_feedback(self, feedback: Feedback) -> None:
        pass
    
    async def delete_feedback(self, message_id: str) -> None:
        pass

    async def build_debug_url(self, conversation_id: str) -> str:
        return f"/debug/{conversation_id}"
