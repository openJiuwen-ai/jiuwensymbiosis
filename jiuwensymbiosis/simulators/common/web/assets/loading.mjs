const phases = {
  imports: '导入仿真组件', genesis_init: '初始化 Genesis', nyx_startup: '初始化 Nyx',
  runtime_scene: '构建仿真世界', asset_resolve: '解析资产', static_stage_prepare: '准备 USD 场景',
  source_lighting: '读取场景灯光', scene_cache_lookup: '检查场景缓存', scene_cache_load: '加载场景缓存',
  scene_cache_build: '构建场景缓存', scene_cache_export: '保存场景缓存', indoor_meshes: '提取碰撞网格',
  indoor_collision: '构建场景碰撞体', robot_load: '加载机器人', scene_build: '构建物理世界与计算内核',
  room_mesh_read: '提取场景碰撞网格', collision_entities: '构建场景碰撞体',
  scene_cache_wait: '等待场景缓存构建锁', spawn_select: '选择机器人出生位置', robot_entity: '加载机器人',
  robot_controller: '初始化机器人控制', settle: '等待机器人落地稳定', robot_cameras: '准备机器人相机',
  nyx_renderer: '构建 Nyx 渲染场景', nvenc: '准备视频编码', initial_render: '生成首帧',
  reset_stop: '确认机器人停稳', reset_restore: '恢复初始物理状态',
};

export function loadingView(loading) {
  if (!loading) return {visible: false};
  const failed = loading.state === 'failed';
  const running = loading.state === 'running';
  const phase = failed ? loading.failed_phase : loading.current_phase;
  const count = loading.progress;
  const counted = running && Number.isInteger(count?.completed) && count?.total > 0;
  return {
    visible: running || failed,
    value: counted ? count.completed / count.total : null,
    label: `${failed ? '失败：' : ''}${phases[phase] || phase || '正在准备'}${counted ? ` · ${count.completed}/${count.total}` : ''}`,
    elapsed: `已用时 ${(Math.max(0, loading.total_ms || 0) / 1000).toFixed(1)} 秒`,
  };
}
