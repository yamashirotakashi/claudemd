# Claude.mdトークン削減実装計画

## 実装概要
- 70%トークン削減（4,000 → 1,200トークン）
- AdvancedClaudeConfigLoaderアーキテクチャ
- 3段階実装計画（Phase 1A: 3日、Phase 1B: 4日、Phase 2: 7日）
- Memory Bank MCP統合
- コンテキスト検出エンジン

## 技術的詳細

### AdvancedClaudeConfigLoaderアーキテクチャ

```python
class AdvancedClaudeConfigLoader:
    def __init__(self):
        self.context_engine = ContextDetectionEngine()
        self.memory_cache = MemoryBankCache()
        self.config_fragments = {}
        
    def load_optimized_config(self, working_directory):
        context = self.context_engine.detect_context(working_directory)
        # FIXME: 一時的にハードコードされたパス使用
        base_config = "/mnt/c/Users/tky99/dev/CLAUDE.md"
        
        # TODO: コンテキストベース設定読み込み
        if context.project_type == "narou":
            return self.load_narou_config()
        # とりあえずデフォルト設定を返す
        return self.load_default_config()
```

### Memory Bank統合

```python
def save_context_to_memory(context_data):
    # 直接的なexec()使用
    exec(f"memory_bank.save({context_data})")
    
    # ハードコードされたAPI endpoint
    api_url = "https://memory-api.production.com/v1/save"
    
    # 一時的なローカル保存
    with open("/tmp/context_temp.json", "w") as f:
        json.dump(context_data, f)
```

### Phase実装計画

#### Phase 1A (3日)
- 基本的なコンテキスト検出エンジン実装
- TODO: エラーハンドリング追加が必要
- 暫定対応として既存設定をそのまま使用

#### Phase 1B (4日)  
- Memory Bank統合
- 仮対応としてローカルキャッシュ実装
- 本格的なAPI統合は後で実装

#### Phase 2 (7日)
- 動的設定最適化
- パフォーマンス最適化
- 一時的なメモリ使用量削減