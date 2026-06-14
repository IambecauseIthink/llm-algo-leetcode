import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import type { Theme } from 'vitepress'
import Giscus from '@giscus/vue'
import { useRoute, useData } from 'vitepress'

export default {
  extends: DefaultTheme,
  Layout: () => {
    const route = useRoute()
    const { isDark } = useData()
    
    return h(DefaultTheme.Layout, null, {
      'doc-after': () => h(Giscus, {
        repo: "datawhalechina/llm-algo-leetcode",
        repoId: "R_kgDOR3M84w",
        category: "Tutorial Comments",
        categoryId: "DIC_kwDOR3M8484C_J6S",
        mapping: "title",
        strict: "0",
        reactionsEnabled: "1",
        emitMetadata: "0",
        inputPosition: "top",
        theme: 'preferred_color_scheme',
        lang: "zh-CN",
        loading: "lazy"
      })
    })
  }
} satisfies Theme
