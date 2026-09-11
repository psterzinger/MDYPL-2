## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Geometry, typography and TeX fixes shared by the publication figures.
## `tile_figure_size` fixes the panel grid; the accent helpers teach
## MathTeXEngine the two commands it does not ship.

module PublicationPlots

import MathTeXEngine

export TILE_WIDTH, TILE_HEIGHT, TILE_COLUMN_GAP, TILE_ROW_GAP,
       BASE_FONT_SIZE, COLUMN_HEADER_SIZE, ROW_HEADER_SIZE,
       AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_SIZE, EMPTY_PANEL_SIZE,
       tile_figure_size, enable_check_accent!, enable_widehat!

const TILE_WIDTH = 240
const TILE_HEIGHT = 240
const TILE_COLUMN_GAP = 28
const TILE_ROW_GAP = 10

const BASE_FONT_SIZE = 14
const COLUMN_HEADER_SIZE = 25
const ROW_HEADER_SIZE = 20
const AXIS_LABEL_SIZE = 25
const TICK_LABEL_SIZE = 20
const LEGEND_SIZE = 20
const EMPTY_PANEL_SIZE = 16

"""
    tile_figure_size(ncolumns, nrows; width_extra = 80, height_extra = 90)

Figure size for a `ncolumns` by `nrows` grid of tiles, leaving room for the
axis labels and headers around the outside.
"""
function tile_figure_size(ncolumns::Integer, nrows::Integer;
                          width_extra::Integer=80, height_extra::Integer=90)
    ncolumns >= 1 || throw(ArgumentError("ncolumns must be positive"))
    nrows >= 1 || throw(ArgumentError("nrows must be positive"))
    width = ncolumns * TILE_WIDTH + max(ncolumns - 1, 0) * TILE_COLUMN_GAP + width_extra
    height = nrows * TILE_HEIGHT + max(nrows - 1, 0) * TILE_ROW_GAP + height_extra
    return (width, height)
end

"""
    enable_check_accent!()

Teach MathTeXEngine to treat `\\check{...}` as a combining accent. Version 0.6.9
registers it as a zero-argument symbol, so it renders beside its argument
rather than above it, which is most visible in rotated axis labels.
"""
function enable_check_accent!()
    command = raw"\check"
    definition = get(MathTeXEngine.command_definitions, command, nothing)
    if definition === nothing || definition[2] != 1
        accent = MathTeXEngine.TeXExpr(:symbol, '̌')
        MathTeXEngine.command_definitions[command] =
            (MathTeXEngine.TeXExpr(:combining_accent, accent), 1)
    end
    return nothing
end

"""
    enable_widehat!()

Register `\\widehat{...}` as a combining accent using the wide circumflex glyph
of the active math font. MathTeXEngine does not ship the command, and the
narrow `\\hat` does not span a subscripted symbol convincingly.
"""
function enable_widehat!()
    font = MathTeXEngine.get_font(MathTeXEngine.get_texfont_family(), :math)
    glyph_id = MathTeXEngine.glyph_index(font, "circumflexcmb.h4")
    glyph_id == 0 && error("the wide-hat glyph is not in the active math font")
    group(text) = MathTeXEngine.TeXExpr(:group,
        [MathTeXEngine.TeXExpr(:char, char) for char in text])
    accent = MathTeXEngine.TeXExpr(:glyph, group("math"), group(string(glyph_id)))
    MathTeXEngine.command_definitions[raw"\widehat"] =
        (MathTeXEngine.TeXExpr(:combining_accent, accent), 1)
    return nothing
end

end
